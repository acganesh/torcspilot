"""
Implement experience replay strategy, known to
improve convergence (see Mnih et al.)
"""

from collections import deque
import random

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buff = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buff, self.num_experiences)
        else:
            return random.sample(self.buff, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buff.append(experience)
            self.num_experiences += 1
        else:
            self.buff.popleft()
            self.buff.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buff = deque()
        self.num_experiences = 0
