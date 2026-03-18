class SupplyChainEnvironment:

    def __init__(self):
        self.inventory = 100
        self.max_inventory = 300
        self.cost = 0

    def step(self, production, shipment, demand):

        # shipment already handled
        shipment = min(shipment, self.inventory)
        self.inventory -= shipment

        satisfied = min(shipment, demand)

        delay = max(0, demand - satisfied)

        # balanced cost
        cost = (
            shipment * 1.5 +         # logistics
            self.inventory * 0.2 +   # holding
            delay * 8               # penalty
        )

        self.cost += cost

        return satisfied, cost