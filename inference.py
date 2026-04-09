"""
inference.py — Baseline inference script for OpenEnv hackathon validator.

Prints structured output to stdout:
  [START] task=NAME
  [STEP] step=N reward=R
  [END] task=NAME score=S steps=N

Environment variables:
    API_BASE_URL — server base URL (default: http://localhost:7860)
    MODEL_NAME   — model name string
    HF_TOKEN     — HuggingFace API token
"""

import os
import sys
import json
import math

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "supply-chain-greedy-v1")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

TASKS = [
    ("easy",   "single_node_supply_chain"),
    ("medium", "multi_node_network"),
    ("hard",   "cascading_disruptions"),
]


# ── Policy ────────────────────────────────────────────────────────────────────

def greedy_policy(obs: dict) -> dict:
    nodes     = obs["nodes"]
    suppliers = obs["suppliers"]

    active = sorted(
        [s for s in suppliers if s["active"]],
        key=lambda s: s["cost_per_unit"]
    )

    if not active:
        return {"action_type": "wait"}

    # Expedite if critical stockout risk
    critical = [n for n in nodes if n["inventory"] < n["demand_per_step"] * 2]
    if critical:
        return {
            "action_type": "expedite",
            "supplier_id": active[0]["id"],
            "quantity":    min(50, max(15, sum(n["demand_per_step"] * 4 for n in critical))),
        }

    # Reroute if severely imbalanced
    by_inv = sorted(nodes, key=lambda n: n["inventory"])
    if len(by_inv) >= 2:
        poor = by_inv[0]
        rich = by_inv[-1]
        if rich["inventory"] > poor["inventory"] * 4 and rich["inventory"] > 30:
            return {
                "action_type":  "reroute",
                "from_node":    rich["id"],
                "to_node":      poor["id"],
                "transfer_qty": 15,
            }

    # Order if running low
    low = [n for n in nodes
           if n["inventory"] < n["demand_per_step"] * 5
           and n["inventory"] < n["capacity"] * 0.5]
    if low:
        return {
            "action_type": "order",
            "supplier_id": active[0]["id"],
            "quantity":    min(60, max(20, sum(n["demand_per_step"] * 6 for n in low))),
        }

    return {"action_type": "wait"}


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(obs: dict, total_reward: float) -> float:
    delivered = obs["total_delivered"]
    stockouts = obs["total_stockouts"]
    total     = delivered + stockouts
    n_nodes   = len(obs["nodes"])
    max_steps = obs["max_steps"]

    svc = delivered / total if total > 0 else 0.0

    exp_cost = max_steps * n_nodes * 60.0 * (1 + n_nodes * 0.1)
    cost_score = math.exp(-obs["total_cost"] / (exp_cost + 1e-9) * 0.5)

    floor   = -(max_steps * n_nodes * 15.0)
    ceiling =   max_steps * n_nodes * 1.5
    rew_score = (total_reward - floor) / (ceiling - floor + 1e-9)

    return round(
        0.4 * max(0.0, min(1.0, svc)) +
        0.3 * max(0.0, min(1.0, cost_score)) +
        0.3 * max(0.0, min(1.0, rew_score)),
        4
    )


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task_direct(difficulty: str, task_name: str) -> dict:
    """Run using the environment directly (no HTTP server needed)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.environment import SupplyChainEnvironment

    env = SupplyChainEnvironment()
    obs = env.reset(difficulty=difficulty, seed=42)

    total_reward = 0.0
    step_num = 0
    done = False

    print(f"[START] task={task_name}", flush=True)

    while not done:
        action = greedy_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_num += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)

    return {
        "task":         task_name,
        "difficulty":   difficulty,
        "score":        score,
        "steps":        step_num,
        "total_reward": round(total_reward, 4),
        "delivered":    obs["total_delivered"],
        "stockouts":    obs["total_stockouts"],
        "total_cost":   obs["total_cost"],
    }


def run_task_http(difficulty: str, task_name: str) -> dict:
    """Run using the HTTP server."""
    import requests
    base = API_BASE_URL.rstrip("/")

    r = requests.post(f"{base}/reset",
                      json={"difficulty": difficulty, "seed": 42},
                      timeout=30)
    r.raise_for_status()
    obs = r.json()["observation"]

    total_reward = 0.0
    step_num = 0
    done = False

    print(f"[START] task={task_name}", flush=True)

    while not done:
        action = greedy_policy(obs)
        r = requests.post(f"{base}/step",
                          json={"action": action},
                          timeout=30)
        r.raise_for_status()
        data = r.json()
        obs   = data["observation"]
        reward = data["reward"]
        done   = data["done"]
        total_reward += reward
        step_num += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)

    return {
        "task":         task_name,
        "difficulty":   difficulty,
        "score":        score,
        "steps":        step_num,
        "total_reward": round(total_reward, 4),
        "delivered":    obs["total_delivered"],
        "stockouts":    obs["total_stockouts"],
        "total_cost":   obs["total_cost"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Decide mode: try HTTP first if API_BASE_URL is set to a remote host,
    # otherwise use direct mode (more reliable for validator)
    use_http = (
        API_BASE_URL != "http://localhost:7860"
        and not API_BASE_URL.startswith("http://localhost")
        and not API_BASE_URL.startswith("http://127.")
    )

    print(f"[INFO] Supply Chain Disruption Manager | model={MODEL_NAME} | mode={'http' if use_http else 'direct'}",
          flush=True)

    results = []
    all_ok  = True

    for difficulty, task_name in TASKS:
        try:
            if use_http:
                result = run_task_http(difficulty, task_name)
            else:
                result = run_task_direct(difficulty, task_name)
        except Exception as e:
            # Fallback to direct mode on HTTP failure
            print(f"[WARN] HTTP failed ({e}), falling back to direct mode", flush=True)
            result = run_task_direct(difficulty, task_name)

        results.append(result)
        if result["score"] < 0.4:
            all_ok = False

    print("[INFO] === FINAL RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    avg = sum(r["score"] for r in results) / len(results)
    print(f"[INFO] average_score={round(avg, 4)}", flush=True)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()