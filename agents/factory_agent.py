class FactoryAgent:

    def __init__(self):
        self.max_capacity = 80

    def act(self, demand, inventory):

        safety_stock = 20
        required = demand + safety_stock - inventory

        return max(10, min(required, self.max_capacity))  