class LogisticsAgent:

    def __init__(self):
        self.capacity = 60

    def act(self, shipment):

        return min(shipment, self.capacity)