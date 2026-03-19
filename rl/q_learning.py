import numpy as np
import random

class QLearningAgent:

    def __init__(self):

        self.actions = [10, 30, 50, 80]

        self.q_table = np.zeros((10, 10, len(self.actions)))

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3

    def discretize(self, value, max_value=300):
        return min(int(value / (max_value / 10)), 9)

    def choose_action(self, inventory, demand):

        i = self.discretize(inventory)
        d = self.discretize(demand)

        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions)-1)

        return np.argmax(self.q_table[i][d])

    def update(self, inv, dem, action, reward, next_inv, next_dem):

        i = self.discretize(inv)
        d = self.discretize(dem)

        ni = self.discretize(next_inv)
        nd = self.discretize(next_dem)

        best_next = np.max(self.q_table[ni][nd])

        self.q_table[i][d][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[i][d][action]
        )