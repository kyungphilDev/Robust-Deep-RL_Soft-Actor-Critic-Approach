import random
import numpy as np

class ExperienceReplay(object):
    def __init__(self, capacity=500000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self,state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)
        #we will it until capacity, then we overwrite
        self.position = ( self.position + 1) % self.capacity
        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
    def sample(self, batch_size):
        #get randomized minibatch
        return zip(*random.sample(self.memory, batch_size))
        
    def __len__(self):
        return len(self.memory)
