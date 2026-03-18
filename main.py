from data_processing.preprocess_pipeline import DataPreprocessor
from forecasting.train_model import DemandModel
from simulation.simulation_runner import run_simulation

def main():

    print("Step 1: Preprocessing Data")
    pipeline = DataPreprocessor("data/raw/demand.csv")
    X_train, X_test, y_train, y_test = pipeline.run()

    print("Step 2: Training Model")
    model = DemandModel()
    model.train(X_train, y_train)

    print("Step 3: Evaluating Model")
    predictions = model.evaluate(X_test, y_test)

    print("Step 4: Running Simulation")
    run_simulation(predictions)

if __name__ == "__main__":
    main()  