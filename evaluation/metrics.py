import numpy as np

def compute_metrics(costs, demands, satisfied):

    total_cost = sum(costs)
    fill_rate = sum(satisfied) / sum(demands)

    delays = [max(0, d - s) for d, s in zip(demands, satisfied)]
    avg_delay = np.mean(delays)

    throughput = sum(satisfied)

    return {
        "Total Cost": total_cost,
        "Fill Rate": fill_rate,
        "Avg Delay": avg_delay,
        "Throughput": throughput
    }