############ import libraries ############

from collections import deque
from operator import itemgetter 
import numpy as np
import random
import torch

#########################################

class ReplayBuffer():
    def __init__(self, maxlen):
        # initialize the replay buffer with a maximum length
        self.buffer = deque(maxlen=maxlen)
    
    def recall(self, batch_size):
        # retrieve a batch of experiences from memory

        # sample a batch of experiences randomly from the buffer
        batch = random.sample(self.buffer, batch_size)

        # unzip the batch into separate variables and stack them into tensors
        state, action_idx, reward, next_state, terminal = map(torch.stack, zip(*batch))

         # squeeze the tensors to remove any singleton dimensions
        state = torch.squeeze(state)
        next_state = torch.squeeze(next_state)
        reward = torch.squeeze(reward)
        terminal = torch.squeeze(terminal)
        return state, action_idx, reward, next_state, terminal

    def append(self, experience):
        # append a new experience to the buffer
        self.buffer.append(experience)

class PrioritizedReplayBuffer():
    def __init__(self, maxlen):
        # initialize the prioritized replay buffer with a maximum length
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def append(self, experience):
        # append a new experience to the buffer and set its priority
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))
        
    def get_probabilities(self, priority_scale):
        # compute the sampling probabilities for each experience based on their priorities
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def get_importance(self, probabilities):
        # compute the importance-sampling weights for the sampled experiences
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized
        
    def recall(self, batch_size, priority_scale=1.0):
        # retrieve a batch of experiences from memory, using prioritized sampling
        sample_size = min(len(self.buffer), batch_size)

        # get sampling probabilities and sample indices based on priorities
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        
        # retrieve the sampled experiences
        batch = itemgetter(*sample_indices)(self.buffer)
        state, action_idx, reward, next_state, terminal = map(torch.stack, zip(*batch))
        state, next_state, reward, terminal= torch.squeeze(state), torch.squeeze(next_state),\
            torch.squeeze(reward), torch.squeeze(terminal)

        # compute importance-sampling weights for the sampled experiences
        importance = torch.from_numpy(self.get_importance(sample_probs[sample_indices]))
        return state, action_idx, reward, next_state, terminal, importance, sample_indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        # update the priorities of the sampled experiences based on TD errors
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
