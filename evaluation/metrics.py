import numpy as np

def compute_metrics(costs, demands, satisfied):

    total_cost = sum(costs)
    # Cap fill rate at 1.0 (100%) - can't satisfy more than 100%
    fill_rate = min(1.0, sum(satisfied) / sum(demands)) if sum(demands) > 0 else 0

    # Delay should measure unmet demand, not excess
    delays = [max(0, d - s) for d, s in zip(demands, satisfied)]
    avg_delay = np.mean(delays) if delays else 0

    throughput = sum(satisfied)

    return {
        "Total Cost": total_cost,
        "Fill Rate": fill_rate,
        "Avg Delay": avg_delay,
        "Throughput": throughput
    }   