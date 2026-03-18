class FactoryAgent:

    def __init__(self):
        self.max_capacity = 80

    def act(self, demand, inventory):

        # produce based on demand gap
        required = demand - inventory

        if required <= 0:
            return 10   # minimal production

        return min(required, self.max_capacity)