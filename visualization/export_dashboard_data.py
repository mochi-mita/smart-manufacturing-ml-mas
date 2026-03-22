import json, os, webbrowser
import shutil
from datetime import datetime
from pathlib import Path


def export_dashboard_data(
    episode_rewards, episode_fill_rates, episode_avg_delays,
    demand_history, satisfied_history, inventory_history,
    production_costs, holding_costs, delay_costs,
    disruption_log, final_metrics, resilience_metrics,
    scenario_comparison, sla,
    agent_events=None,
    rl_meta=None,
    sample_size=600,
    output_html="outputs/dashboard.html",
    open_browser=True,
):
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    type_counts = {}
    for ev in disruption_log:
        type_counts[ev["type"]] = type_counts.get(ev["type"], 0) + 1

    n = min(sample_size, len(demand_history))
    def _r(lst, dp=2): return [round(float(v), dp) for v in lst[-n:]]

    fill_sla  = sla.get("fill_rate", 0.90)
    delay_sla = sla.get("avg_delay", 5.0)

    # Cap agent events for dashboard performance
    events_capped = (agent_events or [])[:600]

    data = {
        "meta": {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "episodes": len(episode_rewards),
            "model": "Q-Learning (tabular 20×20)",
            "dataset": "demand.csv",
            "total_disruption_events": len(disruption_log),
            "total_agent_events": len(agent_events or []),
        },
        "sla": {"fill_rate": fill_sla, "avg_delay": delay_sla},
        "sla_compliance": {
            "fill_rate": {"value": round(float(final_metrics.get("Fill Rate", 0)), 4),
                          "target": fill_sla,
                          "pass": float(final_metrics.get("Fill Rate", 0)) >= fill_sla},
            "avg_delay":  {"value": round(float(final_metrics.get("Avg Delay", 999)), 2),
                           "target": delay_sla,
                           "pass": float(final_metrics.get("Avg Delay", 999)) <= delay_sla},
        },
        "rl_meta": rl_meta or {},
        "final_metrics":      {k: round(float(v), 4) for k, v in final_metrics.items()},
        "resilience_metrics": {k: round(float(v), 4) for k, v in resilience_metrics.items()},
        "scenario_comparison": scenario_comparison,
        "episode_rewards":    [round(float(r), 1) for r in episode_rewards],
        "episode_fill_rates": [round(float(f), 4) for f in episode_fill_rates],
        "episode_avg_delays": [round(float(d), 4) for d in episode_avg_delays],
        "demand_sample":    _r(demand_history),
        "satisfied_sample": _r(satisfied_history),
        "inventory_sample": _r(inventory_history),
        "prod_cost_sample": _r(production_costs),
        "hold_cost_sample": _r(holding_costs),
        "delay_cost_sample":_r(delay_costs),
        "disruption_log":   disruption_log[:200],
        "disruption_type_counts": type_counts,
        "agent_events": events_capped,
    }

    template_path = Path(__file__).parent / "dashboard.html"
    if not template_path.exists():
        print(f"  Warning: dashboard.html template not found at {template_path}")
        with open("outputs/metrics.json", "w") as f:
            json.dump(data, f, indent=2)
        print("  Fallback: data written to outputs/metrics.json")
        return

    template  = template_path.read_text(encoding="utf-8")
    injection = f"window.DASHBOARD_DATA = {json.dumps(data, indent=2)};"
    output    = template.replace("/* __INJECT_DATA__ */", injection)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(output)

    # Auto-copy to repo root as index.html for GitHub Pages
    index_path = Path(__file__).parent.parent / "index.html"
    shutil.copy(output_html, index_path)
    print(f"  GitHub Pages index updated → {index_path}") 

    abs_path = os.path.abspath(output_html)
    url      = "file:///" + abs_path.replace("\\", "/")
    print(f"\nDashboard exported → {output_html}")
    print(f"  Open in browser:  {url}")

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass