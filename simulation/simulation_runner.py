from simulation.environment import SupplyChainEnvironment
from agents.warehouse_agent import WarehouseAgent
from agents.logistics_agent import LogisticsAgent
from evaluation.metrics import compute_metrics
from rl.q_learning import QLearningAgent
from rl.reward_functions import compute_reward
from visualization.plots import plot_learning_curve, plot_demand_vs_supply

def train_rl_agent(predictions, episodes=100):

    rl_agent = QLearningAgent()

    episode_rewards = []

    for ep in range(episodes):

        env = SupplyChainEnvironment()
        warehouse = WarehouseAgent()
        logistics = LogisticsAgent()

        total_reward = 0

        costs = []
        demands = []
        satisfied_list = []

        for day in range(len(predictions)-1):

            demand = predictions[day]
            next_demand = predictions[day+1]

            action_idx = rl_agent.choose_action(env.inventory, demand)
            production = rl_agent.actions[action_idx]

            env.inventory += production

            shipment = warehouse.act(env.inventory, demand)
            transport = logistics.act(shipment)

            satisfied, cost, delay = env.step(0, transport, demand)

            reward = compute_reward(satisfied, demand, cost)

            total_reward += reward

            rl_agent.update(
                env.inventory, demand,
                action_idx, reward,
                env.inventory, next_demand
            )

            rl_agent.epsilon = max(0.01, rl_agent.epsilon * 0.995)

            costs.append(cost)
            demands.append(demand)
            satisfied_list.append(satisfied)

        episode_rewards.append(total_reward)

        print(f"Episode {ep+1} | Reward: {total_reward:.2f}")

    # final metrics
    metrics = compute_metrics(costs, demands, satisfied_list)

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(k, ":", round(v, 2))

    # plots
    plot_learning_curve(episode_rewards)
    plot_demand_vs_supply(demands, satisfied_list)

    return rl_agent