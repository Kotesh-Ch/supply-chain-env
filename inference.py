"""
inference.py — Baseline inference script (REQUIRED by OpenEnv hackathon spec).

This script:
1. Connects to the running server (or instantiates env directly)
2. Runs one episode per difficulty using the greedy agent
3. Prints scores to stdout in the required format

Usage:
    python inference.py
    python inference.py --mode direct          # use env directly (no server needed)
    python inference.py --mode http            # call running server via HTTP

Environment variables:
    API_BASE_URL   — Base URL of the LLM/server API  (default: http://localhost:7860)
    MODEL_NAME     — Model identifier (not used in baseline, but required by spec)
    HF_TOKEN       — Hugging Face token (for gated model access)
"""

import os
import sys
import json
import argparse
import math
from openai import OpenAI  # OpenAI-compatible client as required by spec

# ── Config from env vars ──────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "baseline-greedy")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

# ── OpenAI client (required by spec) ─────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL + "/v1" if not API_BASE_URL.endswith("/v1") else API_BASE_URL,
    api_key=HF_TOKEN or "no-key",
)


# ── Greedy policy (same as graders.py — reproducible) ────────────────────────
"""
def greedy_policy(obs: dict) -> dict:
    nodes     = obs["nodes"]
    suppliers = obs["suppliers"]

    active = sorted(
        [s for s in suppliers if s["active"]],
        key=lambda s: s["cost_per_unit"]
    )

    if not active:
        return {"type": "wait"}

    low_nodes = [n for n in nodes if n["inventory"] < n["demand_per_step"] * 3]

    if low_nodes:
        cheapest = active[0]
        qty = min(40, sum(n["demand_per_step"] * 5 for n in low_nodes))
        return {
            "type": "order",
            "supplier_id": cheapest["id"],
            "quantity": max(10, int(qty)),
        }

    sorted_nodes = sorted(nodes, key=lambda n: n["inventory"])
    if len(sorted_nodes) >= 2:
        poorest = sorted_nodes[0]
        richest = sorted_nodes[-1]
        if richest["inventory"] > poorest["inventory"] * 3 and richest["inventory"] > 20:
            return {
                "type": "reroute",
                "from_node": richest["id"],
                "to_node":   poorest["id"],
                "transfer_qty": 10,
            }

    return {"type": "wait"}
"""
def greedy_policy(obs: dict) -> dict:
    nodes     = obs["nodes"]
    suppliers = obs["suppliers"]

    # Active suppliers sorted by cost
    active = sorted(
        [s for s in suppliers if s["active"]],
        key=lambda s: (s["cost_per_unit"], -s["reliability"])
    )

    if not active:
        return {"type": "wait"}

    supplier = active[0]

    # --- SMART ORDERING ---
    low_nodes = []
    total_order = 0

    for n in nodes:
        demand = n["demand_per_step"]
        inventory = n["inventory"]

        lead_time = supplier["lead_time"]

        # Lookahead + safety buffer
        target = demand * (lead_time + 2)

        if inventory < target:
            needed = target - inventory
            total_order += needed
            low_nodes.append(n)

    if total_order > 0:
        return {
            "type": "order",
            "supplier_id": supplier["id"],
            "quantity": int(min(100, max(20, total_order)))
        }

    # --- SMART REROUTING ---
    sorted_nodes = sorted(nodes, key=lambda n: n["inventory"])

    poorest = sorted_nodes[0]
    richest = sorted_nodes[-1]

    if richest["inventory"] > poorest["inventory"] + 15:
        transfer = min(20, richest["inventory"] // 3)
        return {
            "type": "reroute",
            "from_node": richest["id"],
            "to_node": poorest["id"],
            "transfer_qty": int(transfer),
        }

    return {"type": "wait"}

# ── Direct mode (env instantiated locally) ────────────────────────────────────

def run_direct_mode() -> dict[str, float]:
    """Run inference by importing environment directly."""
    from environment import SupplyChainEnv
    from graders import run_all_graders
    scores = run_all_graders(seed=42)
    return scores


# ── HTTP mode (calls running server) ─────────────────────────────────────────

def run_http_mode() -> dict[str, float]:
    """Run inference by calling the live server endpoints."""
    import requests

    base = API_BASE_URL.rstrip("/")
    scores = {}

    for difficulty in ["easy", "medium", "hard"]:
        # Reset
        r = requests.post(f"{base}/reset", json={"difficulty": difficulty, "seed": 42})
        r.raise_for_status()
        obs = r.json()["observation"]

        total_reward = 0.0
        done = False

        while not done:
            action = greedy_policy(obs)
            r = requests.post(f"{base}/step", json={"action": action})
            r.raise_for_status()
            data = r.json()
            obs    = data["observation"]
            total_reward += data["reward"]
            done   = data["done"]

        # Compute score
        delivered  = obs["total_delivered"]
        stockouts  = obs["total_stockouts"]
        total_cost = obs["total_cost"]
        max_steps  = obs["max_steps"]
        n_nodes    = len(obs["nodes"])

        total_demand = delivered + stockouts
        service_level = delivered / total_demand if total_demand > 0 else 0.0

        expected_cost = max_steps * n_nodes * 50.0
        cost_score = math.exp(-total_cost / (expected_cost + 1e-9))

        floor   = -(max_steps * n_nodes * 10.0)
        ceiling =   max_steps * n_nodes * 1.0
        reward_score = (total_reward - floor) / (ceiling - floor + 1e-9)
        reward_score = max(0.0, min(1.0, reward_score))

        score = round(
            0.4 * max(0.0, min(1.0, service_level)) +
            0.3 * max(0.0, min(1.0, cost_score))    +
            0.3 * max(0.0, min(1.0, reward_score)),
            4
        )
        scores[difficulty] = score
        print(f"[{difficulty.upper():6s}] Delivered={delivered}  "
              f"Stockouts={stockouts}  Cost=${total_cost:.2f}  Score={score:.4f}")

    return scores


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Supply Chain RL Inference Script")
    parser.add_argument("--mode", choices=["direct", "http"], default="direct",
                        help="direct = import env locally | http = call live server")
    args = parser.parse_args()

    print("=" * 55)
    print("  Supply Chain Disruption Manager — Inference")
    print(f"  Mode: {args.mode}  |  Model: {MODEL_NAME}")
    print("=" * 55)

    if args.mode == "http":
        scores = run_http_mode()
    else:
        scores = run_direct_mode()

    print("-" * 55)
    print("FINAL SCORES (JSON):")
    print(json.dumps(scores, indent=2))

    avg = sum(scores.values()) / len(scores)
    print(f"\nAverage: {avg:.4f}")
    print("=" * 55)

    # Exit 0 only if all tasks pass (score >= 0.4)
    all_pass = all(v >= 0.4 for v in scores.values())
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
