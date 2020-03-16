#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import math
import imageio
import os
from scipy.stats import norm


# In[2]:


class Population():
    '''

    '''
    
    def __init__(self, density_map=np.array([]),
                        underlay=None):
        # Initialize height, width of the world
        
        if density_map.shape[0] ==0:
            self.shape = [100,200]
            self.density_map = np.ones((self.shape[0],self.shape[1]))
        else:
            self.shape = [density_map.shape[0],density_map.shape[1]]
            self.density_map = density_map
            self.underlay = underlay
        self.spawn_proba = 0.01
        self.chance_to_stay_within_commute = 0.95
        self.chance_to_move = 0.05
        self.chance_to_commute = 0.04
        self.commute_size_multiplier = 0.015
        self.days_to_recover = 225
        self.infection_chance = norm.pdf(range(self.days_to_recover), int(self.days_to_recover /2), int(self.days_to_recover/4)) * 10
        self.infected_starting_pop = 5
        
        # Init counters
        print(self.infection_chance)
        self.num_infected = 0
        self.num_pre = 0
        self.num_recovered = 0
        
        self.num_infected_arr = []
        self.num_pre_arr = []
        self.num_recovered_arr = []
        
        
        self.init_origins()
    
    def init_origins(self):
        '''
        Initialize a population throughout the world
        given the density of a given area.
        '''
        
        world = []
        for h in range(self.shape[0]):
            for w in range(self.shape[1]):
                
                # This represents how well the population
                # is social distancing according to each individual's abiity
                commute_size = random.random() * min(self.shape)                     * self.commute_size_multiplier
                
                # Initialize the uninfected
                days_until_recovered = 0
                days_left_in_commute = 0
                
                # if spawn chance == success, create a person
                if random.random()*self.density_map[h][w] >.2:
                    world.append([h,w,True,'PRE',commute_size,days_until_recovered,days_left_in_commute])
                    self.num_pre +=1

                # leave the origin as blank
                else:
                    world.append([h,w,False,None,commute_size,days_until_recovered,days_left_in_commute])    
                    
        # Set the 'origin data' as a numpy array            
        self.o_data = np.array(world)
        self.o_data = np.resize(self.o_data,(self.shape[0],self.shape[1],7))
        
        
        # Initialize infected
        for i in range(self.infected_starting_pop):
            valid_init = False
            while valid_init==False:
                h = random.choice(range(self.shape[0]))
                w = random.choice(range(self.shape[1]))

                if random.random()*self.density_map[h][w] >.2:
                    valid_init = True
                    
                commute_size = random.random() * min(self.shape)                     * self.commute_size_multiplier
                days_left_in_commute = 0
            print('Added commute_size of: ',commute_size)
            
            self.o_data[h][w] =                 [h,w,True,'INFECTED',commute_size,self.days_to_recover,days_left_in_commute]
            
            self.num_infected +=1
        print('Finished initialization...')

        return self
    
    def update_infections(self):
        '''
        Check the status of infections, decrement days to recover, 
        and simulate chance to infect based on normal distribution of 
        
        '''
        
        for h in range(self.shape[0]):
            for w in range(self.shape[1]):
                if self.o_data[h][w][2]== True:
                    
                    # If infected, check for neighbors
                    if self.o_data[h][w][3]== 'INFECTED':
                        
                        for hh in range(self.shape[0]):
                            for ww in range(self.shape[1]):
                                
                                # if neighbor is found, roll for infection
                                if self.o_data[hh][ww][2]== True:
                                    
                                    infected_h = self.o_data[h][w][0]
                                    infected_w = self.o_data[h][w][1]
                                    neighbor_h = self.o_data[hh][ww][0]
                                    neighbor_w = self.o_data[hh][ww][1]
                                    
                                    if abs(infected_h-neighbor_h)<=0 and abs(infected_w-neighbor_w)<=0 and self.o_data[hh][ww][3]=='PRE':
                                        
                                        # Lookup probability of infection based on 
                                        # days left to recover.
                                        infection_chance = self.infection_chance[self.o_data[h][w][5]-1]
                                        
                                        roll = random.random()
                                        if roll < infection_chance:
                                            
                                            # Contacted individual has become infected
                                            self.o_data[hh][ww][3] = 'INFECTED'
                                            self.o_data[hh][ww][5] = self.days_to_recover
                                            self.num_infected += 1
                                            self.num_pre -= 1

                        # Decrement days to recover
                        self.o_data[h][w][5] -=1
                        if self.o_data[h][w][5] <=0:
                            self.o_data[h][w][3] = 'RECOVERED'
                            self.o_data[h][w][5] = 0
                            self.num_recovered += 1
                            
                            
                        # Roll for chance to commute
                        if self.o_data[h][w][6] <=0:
                            if random.random() < self.chance_to_commute:
                                self.o_data[h][w][6] = int(50)
                        else:
                            self.o_data[h][w][6] -=1
                    
       
        self.num_infected_arr.append(self.num_infected)
        self.num_pre_arr.append(self.num_pre)
        self.num_recovered_arr.append(self.num_recovered)
        return self
        
    def display_position(self,i):
        '''
        Display the current position of each member of
        the population.
        '''
        
        # Initialize with None
        display_grid = [None for w in range(self.shape[0]) for h in range(self.shape[1])]
        display_grid = np.array(display_grid)
        display_grid = np.resize(display_grid,(self.shape[0],self.shape[1]))
        
        for h in range(self.shape[0]):
            for w in range(self.shape[1]):
                
                
                if self.density_map.shape[0] !=0:
                    # Must be uninverted
                    display_grid[h][w] = self.underlay[h][w] /255
                else:
                    display_grid[h][w] = 0
                
                if self.o_data[h][w][2]== True:

                    # Update display_grid with the current location from origin
                    if self.o_data[h][w][3]== 'INFECTED':
                        display_grid[self.o_data[h][w][0]][self.o_data[h][w][1]] = .0
                    elif self.o_data[h][w][3]== 'PRE':
                        display_grid[self.o_data[h][w][0]][self.o_data[h][w][1]] = 0.5
                    elif self.o_data[h][w][3]== 'RECOVERED':
                        display_grid[self.o_data[h][w][0]][self.o_data[h][w][1]] = .9
                    else:
                        raise AssertionError('Unknown state found')
                      
            
        # Upscale back to 255
        
        for h in range(self.shape[0]):
            for w in range(self.shape[1]):
                display_grid[h][w] = display_grid[h][w] * 255
                    
        aspect =  self.shape[1] /self.shape[0] 
        fig, ax = plt.subplots(figsize=(10,10*aspect))
        ax.imshow(display_grid.astype(float),cmap='gray', vmin=0, vmax=255)
        
        ax.set_title('Infected = {} Healthy = {} Recovered = {}\n'.format(self.num_infected,self.num_pre,self.num_recovered))
        ax.set_axis_off()
        # Used to return the plot as an image array
        fig.savefig('img/{}.png'.format(i))
        plt.close(fig)
        return self
    
    def display_stats(self,steps):
        '''
        Generate stats animation
        '''
        
        fig = plt.figure(figsize(5,10*aspect))
        sns.histogram
        
        return self
    
    def animate(self,steps,fps=None):
        '''
        Render population change over time.
        '''
        
        if not fps:
            fps = 24
        kwargs_write = {'fps':float(fps), 'quantizer':'nq'}
        
        # Take a step, then save the image
        #results = []
        for i in range(steps):
            print('Step {}'.format(i))
            self.take_step()
            print('Took Step')
            self.update_infections()
            print('Updated Infections')
            self.display_position(i)
            print('Generated Image')
 
#         imageio.mimsave('results.gif', results, fps=fps)
        
        png_dir = 'img/'
        images = []
        for i in range(steps):
            file_path = os.path.join(png_dir, '{}.png'.format(i))
            images.append(imageio.imread(file_path))
        imageio.mimsave('result.gif', images, fps=fps)
    
        return self
    
    def take_step(self):
        '''
        Take a random walk as a function of the commute size.
        '''
        
        # For each origin, take a step as a function of the distance
        for h in range(self.shape[0]):
            for w in range(self.shape[1]):
                
                # If commuting or chance_to_move is rolled, then move
                if random.random() < self.chance_to_move or self.o_data[h][w][6] >0:
                    
                    if self.o_data[h][w][2]== True:

                        direction_chance = random.choice(['horiz','vert'])
                        
                        # Keep looping until a valid step is made
                        valid_step = False
                        while valid_step == False:

                            # Calculate distance from origin
                            origin_h = self.o_data[h][w][0]
                            origin_w = self.o_data[h][w][1]

                            # Travel faster if commuting
                            if self.o_data[h][w][6] <=0:
                                if direction_chance=='horiz':
                                    step_h = origin_h+random.choice([1,0,-1])
                                    step_w = origin_w+random.choice([3,0,-3])
                                else:
                                    step_h = origin_h+random.choice([3,0,-3])
                                    step_w = origin_w+random.choice([1,0,-1])
                            else:
                                step_h = origin_h+random.choice([1,0,-1])
                                step_w = origin_w+random.choice([1,0,-1])
                                    
                            # Euclidean distance
                            eucl_dist = math.sqrt(((h - step_h)**2) + ((w - step_w)**2))

                            # If current step exceeds commute size, 
                            # roll for a chance to exceed limit
                            if eucl_dist > self.o_data[h][w][4]:
                                
                                if self.o_data[h][w][6] >0:
                                    # If you are in a commute, you have a distance bonus
                                    if random.random() > self.chance_to_stay_within_commute/10:
                                        valid_step = True
                                elif random.random() > self.chance_to_stay_within_commute:
                                        valid_step = True

                                else:
                                    # Take a step closer to the origin

                                    if (step_h - h) > 0:
                                        step_h = step_h -1
                                    elif (step_h - h) < 0:
                                        step_h = step_h +1


                                    if (step_w - w) > 0:
                                        step_w = step_w -1
                                    elif (step_w - w) < 0:
                                        step_w = step_w +1

                                    valid_step = True
                            else:
                                valid_step = True

                            # Ensure steps are within world
                            if step_h >= self.shape[0] or step_w >= self.shape[1]:
                                valid_step = False

                        # Save step for selected origin
                        self.o_data[h][w][0] = step_h
                        self.o_data[h][w][1] = step_w
        
        return self
                        
                        
    


# In[3]:


# Open density image
import cv2
im = cv2.imread('density.png', cv2.IMREAD_GRAYSCALE)
im_u = cv2.imread('underlay.png', cv2.IMREAD_GRAYSCALE)

resize_factor = 3
im = cv2.resize(im, (int(im.shape[0]/resize_factor),
                    int(im.shape[1]/resize_factor)))
im_u = cv2.resize(im_u, (int(im_u.shape[0]/resize_factor),
                    int(im_u.shape[1]/resize_factor)))

im = im/255
im = 1- im
plt.imshow(im)
plt.imshow(im_u)


# In[ ]:


steps = 2500
p = Population(density_map=im,
              underlay=im_u)
p.animate(steps,fps=24)


# In[5]:


# Export steps to json
import json
df_stats = pd.DataFrame()
df_stats['Infected'] = p.num_infected_arr
df_stats['Healthy'] = p.num_pre_arr
df_stats['Recovered'] = p.num_recovered_arr
df_stats.to_csv('stats_output.csv')


# In[ ]:





# In[ ]:





# In[ ]:




