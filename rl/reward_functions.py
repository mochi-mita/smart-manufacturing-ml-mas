def compute_reward(satisfied, demand, cost):

    service_level = satisfied / (demand + 1e-5)

    reward = (
        service_level * 10      # reward fulfilling demand
        - cost * 0.01           # scaled cost penalty
    )

    return reward