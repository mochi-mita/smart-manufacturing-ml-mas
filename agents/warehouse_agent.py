class WarehouseAgent:

    def act(self, inventory, demand):

        # try to satisfy demand
        return min(inventory, demand) 