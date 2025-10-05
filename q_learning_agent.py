import numpy as np
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, settings):
        self.Q = defaultdict(float)
        self.settings = settings
        
    def get_action(self, state):
        if random.random() < self.settings.epsilon:
            return random.choice(range(4))
        qs = [self.Q[(state, a)] for a in range(4)]
        return int(np.argmax(qs))
        
    def update(self, state, action, reward, next_state):
        best_next_value = max([self.Q[(next_state, a)] for a in range(4)])
        self.Q[(state, action)] += self.settings.alpha * (
            reward + self.settings.gamma * best_next_value - self.Q[(state, action)]
        )