# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:51:25 2023

@author: admin
"""

import numpy as np
import gym
from gym.spaces import Box




class Cruise_Control_Env(gym.Env):
    
    def __init__(self, objective=8., dt=0.001, C=0.02, noise=0.0):
        self.action_space=Box(low=-20, high=20, dtype=np.float32)
        self.observation_space = Box(low=0, high=100,dtype=np.float32)
        self.state = np.array(np.random.randint(5, 10, size=1))
        self.episode_length=100
        self.objective=np.array([objective])
        self.dt=dt
        self.C=C
        self.noise=noise
        
    def step(self, action):
        self.state = self.state + self.dt/self.C *(-self.state+action)
        
        self.episode_length-=1
        
        
        
        reward =  -0.1*(np.absolute(action[0]))-np.linalg.norm(self.state-self.objective)

            
        done=False
        if self.episode_length<=0:
            truncated = True
            
            self.state =self.reset()
        else:
            truncated=False


            self.state=self.state+np.random.randn(1)*self.noise
        
        
        info={}
        
        return self.state, reward, done, truncated, info
    
    def render(self):
        pass
    
    def reset(self):
        self.state = np.array(np.random.randint(5, 10, size=1))
        self.episode_length=100
        return self.state



class Mass_Spring_Damper_Env(gym.Env):
    
    def __init__(self, objective=[ 2.,0.], dt=0.1, c=1,k=5, noise=0.0):
        self.objective=np.array(objective)
        self.observation_space = Box(low=np.array([0, -10]), high=np.array([10, 10]),dtype=np.float32)
        self.action_space=Box(low=-10, high=10, dtype=np.float32)
        self.episode_length=150
        self.dt=dt
        self.c=c
        self.k=k
        self.noise=noise
        self.state = np.random.randint([0,5], [3,7], size=2)
        
        
    def step(self, action):
        x = self.state[0]
        v = self.state[1]
        xnext = x + self.dt * v
        vnext = v + self.dt*(action - self.c*v-self.k*x)
        self.state = np.array([xnext, vnext[0]], dtype=np.float32)
        self.episode_length-=1
        
        reward =  -0.1*(np.absolute(action[0]))-np.linalg.norm(xnext-self.objective[0])-np.absolute(vnext[0])
        
        

        
        done=False
        if self.episode_length<=0:
            truncated = True
            
            self.state =self.reset()
        else:
            truncated=False


            self.state=self.state+np.random.randn(1)*self.noise
        
        
        info={}
        
        return self.state, reward, done, truncated, info
    
    def render(self):
        pass
    
    def reset(self):
        self.state = np.random.randint([0,5], [3,7], size=2)
        self.episode_length=150
        return self.state


class Custom_2nd_order_Env(gym.Env):
    
    def __init__(self, matrix, objective=[ 0.,0.], dt=0.05, noise=0.0):
        self.objective=np.array(objective)
        self.observation_space = Box(low=np.array([-100, -100]), high=np.array([100, 100]),dtype=np.float32)
        self.action_space=Box(low=-20, high=20, dtype=np.float32)
        self.episode_length=200
        self.dt=dt
        self.a=matrix[0,0]
        self.b=matrix[0,1]
        self.c=matrix[1,0]
        self.d=matrix[1,1]
        self.noise=noise
        self.state = np.random.rand(2)
        
        
    def step(self, action):
        x = self.state[0]
        # print(x)
        v = self.state[1]
        xnext = x + self.dt * (self.a*x+ self.b*v)
        vnext = v + self.dt*(action + self.c*x + self.d*v)
        self.state = np.array([xnext, vnext[0]], dtype=np.float32)
        self.episode_length-=1
        
        reward =  -0.1*(np.absolute(action[0]))-np.linalg.norm(xnext-self.objective[0])-np.linalg.norm(vnext-self.objective[1])

        
        done=False
        if self.episode_length<=0:
            truncated = True
            
            self.state =self.reset()
        else:
            truncated=False


            self.state=self.state+np.random.randn(1)*self.noise
        
        
        info={}
        
        return self.state, reward, done, truncated, info
    
    def render(self):
        pass
    
    def reset(self):
        self.state = np.random.rand(2)
        self.episode_length=200
        return self.state

