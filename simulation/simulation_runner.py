from simulation.environment import SupplyChainEnvironment
from agents.warehouse_agent import WarehouseAgent
from agents.logistics_agent import LogisticsAgent
from evaluation.metrics import compute_metrics
from rl.q_learning import QLearningAgent

def run_simulation(predictions):

    env = SupplyChainEnvironment()

    warehouse = WarehouseAgent()
    logistics = LogisticsAgent()

    rl_agent = QLearningAgent()

    costs = []
    demands = []
    satisfied_list = []

    for day in range(len(predictions)-1):

        demand = predictions[day]
        next_demand = predictions[day+1]

        # RL decides production
        action_idx = rl_agent.choose_action(env.inventory, demand)
        production = rl_agent.actions[action_idx]

        env.inventory += production

        shipment = warehouse.act(env.inventory, demand)
        transport = logistics.act(shipment)

        satisfied, cost = env.step(0, transport, demand)
        
        # Ensure satisfied is per-day, not cumulative
        satisfied = min(satisfied, demand)  # Can't satisfy more than demand

        reward = satisfied - cost

        rl_agent.update(
            env.inventory, demand,
            action_idx, reward,
            env.inventory, next_demand
        )

        rl_agent.epsilon = max(0.05, rl_agent.epsilon * 0.995)

        costs.append(cost)
        demands.append(demand)
        satisfied_list.append(satisfied)

        print(f"Day {day} | Demand:{demand:.2f} | Inv:{env.inventory} | Satisfied:{satisfied:.2f} | Cost:{cost:.2f}")

    metrics = compute_metrics(costs, demands, satisfied_list)

    print("\nFinal Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {round(v, 2)}")