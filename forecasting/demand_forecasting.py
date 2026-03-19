from sklearn.ensemble import RandomForestRegressor

class DemandForecaster:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X) 