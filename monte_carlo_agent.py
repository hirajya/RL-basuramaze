import numpy as np
import random
from collections import defaultdict

class MonteCarloAgent:
    def __init__(self, settings):
        self.Q = defaultdict(float)
        self.Returns = defaultdict(list)
        self.settings = settings
        
    def get_action(self, state):
        if random.random() < self.settings.epsilon:
            return random.choice(range(4))
        qs = [self.Q[(state, a)] for a in range(4)]
        return int(np.argmax(qs))
        
    def update(self, episode_data):
        G = 0
        for (state, action, reward) in reversed(episode_data):
            G = self.settings.gamma * G + reward
            if not any((x[0] == state and x[1] == action) for x in episode_data[:-1]):
                self.Returns[(state, action)].append(G)
                self.Q[(state, action)] = np.mean(self.Returns[(state, action)])