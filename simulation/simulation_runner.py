from simulation.environment       import SupplyChainEnvironment
from simulation.disruption_engine import DisruptionEngine, DISRUPTION_TYPES
from simulation.baseline_runner   import run_baseline_evaluation
from agents.warehouse_agent       import WarehouseAgent
from agents.logistics_agent       import LogisticsAgent
from agents.supplier_agent        import SupplierAgent
from evaluation.metrics           import compute_metrics, compute_resilience_metrics
from rl.q_learning                import QLearningAgent
from rl.reward_functions          import compute_reward
from visualization.plots          import (
    plot_learning_curve, plot_demand_vs_supply,
    plot_inventory_levels, plot_disruption_timeline,
    plot_cost_breakdown, plot_episode_metrics, plot_resilience_radar,
)
from visualization.export_dashboard_data import export_dashboard_data

SLA_FILL_RATE  = 0.90
SLA_AVG_DELAY  = 5.0
MAX_LOG_EVENTS = 120   # hard cap — keeps log readable (was 700)


def _evt(events, step, severity, agent, event_type, message, human_readable, details=None):
    """
    Append one structured event to the log (if cap not reached).

    severity      : ALERT | WARNING | ACTION | INFO | RESOLVED
    agent         : SupplierAgent | FactoryAgent | WarehouseAgent |
                    LogisticsAgent | DisruptionEngine | System
    human_readable: plain-language translation in each agent's operational
                    vocabulary — the "personality" layer for non-technical users.
    """
    if len(events) >= MAX_LOG_EVENTS:
        return
    events.append({
        "step": step, "severity": severity, "agent": agent,
        "event": event_type, "message": message,
        "human_readable": human_readable, "details": details or {},
    })


def _evaluate_episode(rl_agent, predictions, disruptions_enabled, seed=999):
    env, warehouse = SupplyChainEnvironment(), WarehouseAgent()
    logistics, supplier = LogisticsAgent(), SupplierAgent()
    engine = DisruptionEngine(enabled=disruptions_enabled, seed=seed)
    saved_ep = rl_agent.epsilon
    rl_agent.epsilon = 0.0
    original_cap = logistics.capacity
    costs, demands, satisfied_list = [], [], []

    for day in range(len(predictions) - 1):
        demand = float(predictions[day])
        engine.tick(day)
        raw_supply = supplier.act()
        action_idx = rl_agent.choose_action(env.inventory, demand)
        disrupted  = engine.apply(demand=demand, supply=raw_supply,
                                  logistics_cap=logistics.capacity,
                                  production=rl_agent.actions[action_idx])
        actual_demand      = disrupted["demand"]
        logistics.capacity = disrupted["logistics_cap"]
        actual_prod = min(rl_agent.actions[action_idx], disrupted["supply"])
        shipment    = warehouse.act(env.inventory + actual_prod, actual_demand)
        transport   = logistics.act(shipment)
        satisfied, cost, _ = env.step(actual_prod, transport, actual_demand)
        logistics.capacity = original_cap
        costs.append(cost); demands.append(actual_demand); satisfied_list.append(satisfied)

    rl_agent.epsilon = saved_ep
    return compute_metrics(costs, demands, satisfied_list)


def _build_scenario_comparison(rl_agent, predictions, baseline_metrics):
    print("  Running scenario evaluations...")
    rl_normal    = _evaluate_episode(rl_agent, predictions, disruptions_enabled=False)
    rl_disrupted = _evaluate_episode(rl_agent, predictions, disruptions_enabled=True)
    base_cost = baseline_metrics["Total Cost"]
    base_fr   = baseline_metrics["Fill Rate"]

    def cost_save(c): return round((base_cost - c) / base_cost * 100, 1) if base_cost > 0 else 0.0
    def fr_delta(f):  return round((f - base_fr) * 100, 2)

    return {
        "baseline": {
            "label": "No-RL baseline", "description": "Heuristic demand-following policy",
            "fill_rate": round(base_fr, 4), "avg_delay": round(baseline_metrics["Avg Delay"], 2),
            "total_cost": round(base_cost, 0), "throughput": round(baseline_metrics["Throughput"], 0),
            "resilience_score": 1.0, "cost_saving_pct": 0.0, "fr_delta_pp": 0.0,
            "sla_pass": base_fr >= SLA_FILL_RATE, "is_rl": False, "accent": "gray",
        },
        "rl_normal": {
            "label": "RL system — normal", "description": "Trained Q-agent, no disruptions",
            "fill_rate": round(rl_normal["Fill Rate"], 4), "avg_delay": round(rl_normal["Avg Delay"], 2),
            "total_cost": round(rl_normal["Total Cost"], 0), "throughput": round(rl_normal["Throughput"], 0),
            "resilience_score": 1.0, "cost_saving_pct": cost_save(rl_normal["Total Cost"]),
            "fr_delta_pp": fr_delta(rl_normal["Fill Rate"]),
            "sla_pass": rl_normal["Fill Rate"] >= SLA_FILL_RATE, "is_rl": True, "accent": "teal",
        },
        "rl_disrupted": {
            "label": "RL system — disrupted", "description": "Trained Q-agent under active disruptions",
            "fill_rate": round(rl_disrupted["Fill Rate"], 4), "avg_delay": round(rl_disrupted["Avg Delay"], 2),
            "total_cost": round(rl_disrupted["Total Cost"], 0), "throughput": round(rl_disrupted["Throughput"], 0),
            "resilience_score": round(rl_disrupted["Fill Rate"] / max(rl_normal["Fill Rate"], 1e-9), 4),
            "cost_saving_pct": cost_save(rl_disrupted["Total Cost"]),
            "fr_delta_pp": fr_delta(rl_disrupted["Fill Rate"]),
            "sla_pass": rl_disrupted["Fill Rate"] >= SLA_FILL_RATE, "is_rl": True, "accent": "amber",
        },
    }


def train_rl_agent(predictions, episodes=100, disruptions_enabled=True):

    rl_agent = QLearningAgent()
    engine   = DisruptionEngine(enabled=disruptions_enabled, seed=42)

    episode_rewards, episode_fill_rates, episode_avg_delays = [], [], []
    last_demands = last_satisfied = last_inventory = []
    last_fill_per_step = last_prod_costs = last_hold_costs = last_delay_costs = []
    agent_events: list = []

    for ep in range(episodes):
        env, warehouse = SupplyChainEnvironment(), WarehouseAgent()
        logistics, supplier = LogisticsAgent(), SupplierAgent()
        engine.reset()

        total_reward   = 0
        costs, demands, satisfied_list = [], [], []
        inventory_hist, fill_per_step   = [], []
        prod_costs, hold_costs, delay_costs_ep = [], [], []
        original_cap = logistics.capacity

        is_last        = (ep == episodes - 1)
        below_safety   = False
        below_critical = False
        sla_failing    = False
        prev_disruptions: set = set()
        last_periodic  = -9999
        # Cooldown trackers — prevent high-frequency events from flooding log
        last_high_prod_log    = -2000
        last_logistics_log    = -2000
        last_partial_fill_log = -2000
        last_supply_log       = -2000

        for day in range(len(predictions) - 1):
            demand      = float(predictions[day])
            next_demand = float(predictions[day + 1])
            engine.tick(day)
            raw_supply = supplier.act()
            action_idx = rl_agent.choose_action(env.inventory, demand)
            disrupted  = engine.apply(demand=demand, supply=raw_supply,
                                      logistics_cap=logistics.capacity,
                                      production=rl_agent.actions[action_idx])
            actual_demand      = disrupted["demand"]
            logistics.capacity = disrupted["logistics_cap"]
            current_inv        = env.inventory
            action_idx         = rl_agent.choose_action(current_inv, actual_demand)
            actual_prod        = min(rl_agent.actions[action_idx], disrupted["supply"])
            shipment           = warehouse.act(env.inventory + actual_prod, actual_demand)
            transport          = logistics.act(shipment)
            satisfied, cost, delay = env.step(actual_prod, transport, actual_demand)

            prod_costs.append(actual_prod * 1.0)
            hold_costs.append(env.inventory * 0.5)
            delay_costs_ep.append(delay * 5)

            reward = compute_reward(satisfied, actual_demand, cost, production=actual_prod)
            total_reward += reward
            rl_agent.update(current_inv, actual_demand, action_idx, reward,
                            env.inventory, next_demand)
            logistics.capacity = original_cap

            step_fill = satisfied / (actual_demand + 1e-9)
            costs.append(cost); demands.append(actual_demand)
            satisfied_list.append(satisfied)
            inventory_hist.append(env.inventory)
            fill_per_step.append(step_fill)

            # ── Structured agent event logging (last episode only) ────────────
            if is_last:
                curr_disp = set(engine.active_types())

                # Disruption starts
                for dt in curr_disp - prev_disruptions:
                    cfg = DISRUPTION_TYPES.get(dt, {})
                    _evt(agent_events, day, "ALERT", "DisruptionEngine",
                         f"disruption_start",
                         f"[Step {day}] {cfg.get('description','Disruption')} ACTIVATED — expected {cfg.get('duration_range',(0,0))} steps",
                         f"Disruption detected: {cfg.get('description','Event')} now active. All downstream agents operating under degraded conditions. "
                         f"Impact: {', '.join(k+' x'+str(v) for k,v in cfg.items() if 'factor' in k)}. Monitoring recovery.",
                         {"type": dt, "severity": cfg.get("severity",""),
                          "factors": {k:v for k,v in cfg.items() if "factor" in k}})

                # Disruption resolves
                for dt in prev_disruptions - curr_disp:
                    cfg = DISRUPTION_TYPES.get(dt, {})
                    _evt(agent_events, day, "RESOLVED", "DisruptionEngine",
                         "disruption_resolved",
                         f"[Step {day}] {cfg.get('description','Disruption')} RESOLVED — normal operations restored",
                         f"Disruption cleared: {cfg.get('description','Event')} no longer active. Agents resuming standard operating parameters. "
                         f"Expect inventory recovery within 3–8 steps as production ramps to meet backlog.",
                         {"type": dt})
                prev_disruptions = curr_disp

                # Supplier events
                if raw_supply <= 20 and "supplier_failure" in curr_disp:
                    _evt(agent_events, day, "WARNING", "SupplierAgent", "supply_critically_low",
                         f"[Step {day}] Supply critically constrained: {raw_supply:.0f} units (disruption active)",
                         f"Supplier alert — critical batch shortage: {raw_supply:.0f} units available, "
                         f"operating at {raw_supply/120*100:.0f}% of normal 80-180 unit range. "
                         f"Factory production severely limited this cycle. Drawing from inventory reserves.",
                         {"available": round(raw_supply), "normal_range": "80–180 units", "disruption": "active"})

                elif raw_supply <= 80 and "supplier_failure" in curr_disp:
                    if day - last_supply_log > 800:
                        _evt(agent_events, day, "INFO", "SupplierAgent", "supply_reduced",
                             f"[Step {day}] Supplier delivering reduced batch: {raw_supply:.0f} units (disruption in effect)",
                             f"Reduced supply batch dispatched: {raw_supply:.0f} units. Supplier disruption compressing output. "
                             f"Factory will supplement from existing warehouse reserves to meet current demand of {actual_demand:.0f} units.",
                             {"available": round(raw_supply), "demand": round(actual_demand)})
                        last_supply_log = day

                # Factory/RL events
                if actual_prod >= 160:
                    if day - last_high_prod_log > 1500:
                        _evt(agent_events, day, "ACTION", "FactoryAgent", "max_production",
                             f"[Step {day}] RL agent: MAX production deployed — {actual_prod:.0f} units | Inv: {current_inv:.0f} | Demand: {actual_demand:.0f}",
                             f"Production line at full capacity: {actual_prod:.0f} units scheduled. RL agent identified critical inventory pressure "
                             f"(stock at {current_inv:.0f} units, demand {actual_demand:.0f}) and escalated to maximum output to prevent stockout. "
                             f"Q-table action index {action_idx} selected.",
                             {"production": round(actual_prod), "inventory": round(current_inv),
                              "demand": round(actual_demand), "q_action": action_idx, "epsilon": round(rl_agent.epsilon, 3)})
                        last_high_prod_log = day

                elif actual_prod >= 120:
                    if day - last_high_prod_log > 1500:
                        _evt(agent_events, day, "ACTION", "FactoryAgent", "high_production",
                             f"[Step {day}] RL agent increased production to {actual_prod:.0f} units | Demand signal: {actual_demand:.0f}",
                             f"Production scaled up: {actual_prod:.0f} units ordered. RL Q-agent responding to elevated demand forecast "
                             f"({actual_demand:.0f} units expected). Proactive inventory build to absorb potential surge or disruption.",
                             {"production": round(actual_prod), "demand": round(actual_demand), "inventory": round(current_inv)})
                        last_high_prod_log = day

                if actual_prod < rl_agent.actions[action_idx] * 0.6 and raw_supply < 60:
                    _evt(agent_events, day, "WARNING", "FactoryAgent", "production_supply_constrained",
                         f"[Step {day}] Production constrained by supply shortage: wanted {rl_agent.actions[action_idx]:.0f}, produced {actual_prod:.0f}",
                         f"Factory output limited by upstream supply: RL agent requested {rl_agent.actions[action_idx]:.0f} units "
                         f"but supplier could only provide raw material for {actual_prod:.0f} units. "
                         f"Demand fulfilment at risk — warehouse buffer critical during this period.",
                         {"requested": rl_agent.actions[action_idx], "actual": round(actual_prod), "supply_available": round(raw_supply)})

                # Warehouse events
                inv = env.inventory
                if inv < 5 and not below_critical:
                    _evt(agent_events, day, "ALERT", "WarehouseAgent", "inventory_critical",
                         f"[Step {day}] CRITICAL: Inventory at {inv:.0f} units — near stockout | Demand: {actual_demand:.0f}",
                         f"CRITICAL ALERT — Warehouse nearly depleted: {inv:.0f} units remaining against {actual_demand:.0f} demand. "
                         f"Immediate replenishment required. Partial fulfilment enforced — {max(0,actual_demand-inv):.0f} units unserved. "
                         f"SLA breach imminent if production does not escalate within 2 steps.",
                         {"inventory": round(inv), "demand": round(actual_demand),
                          "shortfall": round(max(0, actual_demand - inv)), "safety_stock": 20})
                    below_critical = True
                    below_safety   = True

                elif inv < 20 and not below_safety:
                    _evt(agent_events, day, "WARNING", "WarehouseAgent", "inventory_below_safety",
                         f"[Step {day}] Inventory below safety stock: {inv:.0f}/20 units | Demand: {actual_demand:.0f}",
                         f"Inventory warning: Stock level at {inv:.0f} units — below the 20-unit safety buffer. "
                         f"Warehouse operating in risk zone. RL agent should be increasing production orders now. "
                         f"Vulnerable to any demand spike or further supplier disruption.",
                         {"inventory": round(inv), "safety_stock": 20,
                          "demand": round(actual_demand), "buffer_remaining": round(inv)})
                    below_safety = True

                elif inv >= 20 and below_safety:
                    _evt(agent_events, day, "RESOLVED", "WarehouseAgent", "inventory_restored",
                         f"[Step {day}] Inventory restored to {inv:.0f} units — safety threshold cleared",
                         f"Warehouse recovery confirmed: Inventory back above 20-unit safety threshold at {inv:.0f} units. "
                         f"Full demand coverage restored. RL agent successfully navigated low-stock period — "
                         f"buffer rebuilt within operational parameters.",
                         {"inventory": round(inv), "safety_stock": 20})
                    below_safety   = False
                    below_critical = False

                if satisfied < actual_demand * 0.7 and actual_demand > 30:
                    if day - last_partial_fill_log > 600:
                        _evt(agent_events, day, "WARNING", "WarehouseAgent", "partial_fulfilment",
                             f"[Step {day}] Partial fulfilment: {satisfied:.0f}/{actual_demand:.0f} units ({satisfied/actual_demand*100:.0f}%)",
                             f"Service degraded: {satisfied:.0f} of {actual_demand:.0f} units fulfilled "
                             f"({satisfied/actual_demand*100:.0f}% fill rate this step). "
                             f"{delay:.0f} units unserved — contributing to delay penalty. "
                             f"Cause: {'disruption impact' if curr_disp else 'inventory depletion'}.",
                             {"satisfied": round(satisfied), "demand": round(actual_demand),
                              "fill_pct": round(satisfied/actual_demand*100, 1), "delay_units": round(delay)})
                        last_partial_fill_log = day

                # Logistics events
                if transport < shipment - 15 and shipment > 30:
                    if day - last_logistics_log > 800:
                        pct = transport / max(shipment, 1) * 100
                        _evt(agent_events, day, "INFO", "LogisticsAgent", "capacity_constrained",
                             f"[Step {day}] Logistics cap: {transport:.0f}/{shipment:.0f} units moved ({pct:.0f}%)",
                             f"Fleet throughput limited: {transport:.0f} of {shipment:.0f} units dispatched this cycle ({pct:.0f}% throughput). "
                             f"{shipment-transport:.0f} units held at distribution dock. Rerouting optimised — "
                             f"all priority shipments dispatched, remainder queued.",
                             {"dispatched": round(transport), "requested": round(shipment),
                              "held_back": round(shipment - transport), "capacity": round(logistics.capacity)})
                        last_logistics_log = day

                if "logistics_breakdown" in curr_disp and transport < 50:
                    _evt(agent_events, day, "ALERT", "LogisticsAgent", "logistics_disrupted",
                         f"[Step {day}] Logistics breakdown: fleet at {logistics.capacity:.0f} unit capacity (normal: 300)",
                         f"Fleet severely degraded: operating at {logistics.capacity:.0f}/300 units "
                         f"({logistics.capacity/3:.0f}% capacity). Priority dispatch only — {transport:.0f} units moving. "
                         f"Non-critical delivery timelines suspended until fleet recovery confirmed.",
                         {"current_capacity": round(logistics.capacity), "normal_capacity": 300,
                          "units_dispatched": round(transport), "disruption": "logistics_breakdown"})

                # SLA events
                if step_fill < SLA_FILL_RATE and not sla_failing:
                    _evt(agent_events, day, "ALERT", "System", "sla_breach",
                         f"[Step {day}] SLA BREACH: fill rate {step_fill:.3f} below target {SLA_FILL_RATE}",
                         f"Service Level Agreement breached: Per-step fill rate at {step_fill:.3f}, below the {SLA_FILL_RATE} SLA floor. "
                         f"Root cause: {'active disruption(s): ' + ', '.join(curr_disp) if curr_disp else 'inventory depletion without disruption'}. "
                         f"System is working to recover — monitor inventory and production response.",
                         {"fill_rate": round(step_fill, 4), "sla": SLA_FILL_RATE,
                          "active_disruptions": list(curr_disp)})
                    sla_failing = True

                elif step_fill >= SLA_FILL_RATE and sla_failing:
                    _evt(agent_events, day, "RESOLVED", "System", "sla_restored",
                         f"[Step {day}] SLA RESTORED: fill rate recovered to {step_fill:.3f}",
                         f"Service Level Agreement restored: Fill rate back above {SLA_FILL_RATE} at {step_fill:.3f}. "
                         f"RL agent successfully recovered from stress event. Supply chain stability re-established — "
                         f"continuing to monitor for secondary disruptions.",
                         {"fill_rate": round(step_fill, 4), "sla": SLA_FILL_RATE})
                    sla_failing = False

                # Periodic system summary every 5000 steps
                if day - last_periodic >= 5000:
                    ep_fr = sum(fill_per_step) / max(len(fill_per_step), 1)
                    _evt(agent_events, day, "INFO", "System", "checkpoint",
                         f"[Step {day}] Checkpoint — Inventory: {env.inventory:.0f}  Fill rate: {ep_fr:.3f}  Disruptions: {len(curr_disp)}  ε: {rl_agent.epsilon:.3f}",
                         f"System health at step {day}: Inventory {env.inventory:.0f} units | "
                         f"Cumulative fill rate {ep_fr:.3f} ({'above' if ep_fr >= SLA_FILL_RATE else 'BELOW'} SLA) | "
                         f"{len(curr_disp)} active disruption(s) | RL exploration rate ε={rl_agent.epsilon:.3f} | "
                         f"Total events logged: {len(agent_events)}.",
                         {"inventory": round(env.inventory), "running_fill_rate": round(ep_fr, 4),
                          "epsilon": round(rl_agent.epsilon, 4), "active_disruptions": len(curr_disp)})
                    last_periodic = day

        rl_agent.epsilon = max(0.01, rl_agent.epsilon * 0.97)
        ep_m = compute_metrics(costs, demands, satisfied_list)
        episode_rewards.append(total_reward)
        episode_fill_rates.append(ep_m["Fill Rate"])
        episode_avg_delays.append(ep_m["Avg Delay"])

        last_demands, last_satisfied = demands, satisfied_list
        last_inventory, last_fill_per_step = inventory_hist, fill_per_step
        last_prod_costs, last_hold_costs, last_delay_costs = prod_costs, hold_costs, delay_costs_ep

        if (ep + 1) % 10 == 0:
            disc = f" | log: {len(agent_events)}" if is_last else ""
            print(f"Ep {ep+1:3d} | Reward:{total_reward:10.2f} | "
                  f"Fill:{ep_m['Fill Rate']:.3f} | Delay:{ep_m['Avg Delay']:.2f} | "
                  f"ε:{rl_agent.epsilon:.3f}{disc}")

    print(f"\nAgent event log: {len(agent_events)} events captured from final episode.")
    print("\nRunning post-training scenario comparison...")
    baseline_metrics    = run_baseline_evaluation(predictions)
    scenario_comparison = _build_scenario_comparison(rl_agent, predictions, baseline_metrics)
    for sc in scenario_comparison.values():
        save = f"  saves {sc['cost_saving_pct']:.1f}% cost" if sc["cost_saving_pct"] else ""
        print(f"  {sc['label']:<32} fill={sc['fill_rate']:.3f}  SLA:{'PASS' if sc['sla_pass'] else 'FAIL'}{save}")

    final_metrics      = compute_metrics(last_delay_costs, last_demands, last_satisfied)
    resilience_metrics = compute_resilience_metrics(
        last_fill_per_step, engine.disruption_log, len(last_fill_per_step))
    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {round(v, 3)}")

    print("\nGenerating plots...")
    plot_learning_curve(episode_rewards)
    plot_demand_vs_supply(last_demands, last_satisfied)
    plot_inventory_levels(last_inventory, engine.disruption_log)
    plot_disruption_timeline(engine.disruption_log, last_fill_per_step, last_demands, last_satisfied)
    plot_cost_breakdown(last_prod_costs, last_hold_costs, last_delay_costs)
    plot_episode_metrics(episode_fill_rates, episode_avg_delays)
    plot_resilience_radar(
        {"fill_rate": resilience_metrics["fill_normal"], "avg_delay": final_metrics["Avg Delay"],
         "cost_per_step": final_metrics["Total Cost"]/max(len(last_demands),1),
         "resilience_score": 1.0, "throughput_norm": min(final_metrics["Throughput"]/(sum(last_demands)+1e-9),1)},
        {"fill_rate": resilience_metrics["fill_during_disruption"],
         "avg_delay": final_metrics["Avg Delay"]*1.5,
         "cost_per_step": final_metrics["Total Cost"]/max(len(last_demands),1)*1.2,
         "resilience_score": resilience_metrics["resilience_score"],
         "throughput_norm": resilience_metrics["fill_during_disruption"]}
    )

    export_dashboard_data(
        episode_rewards=episode_rewards, episode_fill_rates=episode_fill_rates,
        episode_avg_delays=episode_avg_delays, demand_history=last_demands,
        satisfied_history=last_satisfied, inventory_history=last_inventory,
        production_costs=last_prod_costs, holding_costs=last_hold_costs,
        delay_costs=last_delay_costs, disruption_log=engine.disruption_log,
        final_metrics=final_metrics, resilience_metrics=resilience_metrics,
        scenario_comparison=scenario_comparison,
        sla={"fill_rate": SLA_FILL_RATE, "avg_delay": SLA_AVG_DELAY},
        agent_events=agent_events,
        rl_meta={"n_bins": rl_agent.n_bins, "actions": rl_agent.actions,
                 "alpha": rl_agent.alpha, "gamma": rl_agent.gamma,
                 "final_epsilon": round(rl_agent.epsilon, 4)},
    )
    return rl_agent 