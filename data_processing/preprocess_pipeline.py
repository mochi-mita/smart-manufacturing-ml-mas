import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath)

    def convert_datetime(self):
        self.df["date"] = pd.to_datetime(self.df["date"])

    def sort_data(self):
        self.df = self.df.sort_values(["store","item","date"])

    def create_time_features(self):
        self.df["day"] = self.df["date"].dt.day
        self.df["month"] = self.df["date"].dt.month
        self.df["year"] = self.df["date"].dt.year
        self.df["day_of_week"] = self.df["date"].dt.dayofweek

    def create_lag_features(self):
        self.df["lag_1"] = self.df.groupby(["store","item"])["sales"].shift(1)
        self.df["lag_7"] = self.df.groupby(["store","item"])["sales"].shift(7)

    def handle_missing(self):
        self.df = self.df.bfill()
        self.df = self.df.ffill()

    def select_features(self):
        features = [
            "store","item","day","month","year",
            "day_of_week","lag_1","lag_7"
        ]
        X = self.df[features]
        y = self.df["sales"]
        return X, y

    def split_data(self, X, y):
        return train_test_split(X, y, test_size=0.2, shuffle=False)

    def save_processed(self):
        output_path = "data/processed/demand_processed.csv"
        self.df.to_csv(output_path, index=False)
        print(f"Processed data saved at: {output_path}")

    def run(self):
        self.load_data()
        self.convert_datetime()
        self.sort_data()
        self.create_time_features()
        self.create_lag_features()
        self.handle_missing()

        self.save_processed()

        X, y = self.select_features()
        return self.split_data(X, y)
 