import matplotlib.pyplot as plt
import os

def plot_learning_curve(rewards):

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure()
    plt.plot(rewards)
    plt.title("RL Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid()

    plt.savefig("outputs/plots/learning_curve.png")
    plt.close()


def plot_demand_vs_supply(demand, satisfied):

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure()
    plt.plot(demand, label="Demand")
    plt.plot(satisfied, label="Satisfied")
    plt.legend()
    plt.title("Demand vs Supply")

    plt.savefig("outputs/plots/demand_vs_supply.png")
    plt.close()