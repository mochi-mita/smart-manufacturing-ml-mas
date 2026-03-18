from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

class DemandModel:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=50)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model MAE: {mae:.2f}")
        return preds