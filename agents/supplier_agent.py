import random

class SupplierAgent:

    def __init__(self):
        self.supply_levels = [30, 50, 70]

    def act(self):
        return random.choice(self.supply_levels)