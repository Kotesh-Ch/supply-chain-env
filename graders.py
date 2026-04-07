"""
graders.py — Agent graders for all 3 tasks (easy / medium / hard).
Each grader runs an episode and returns a score in [0.0, 1.0].
"""

from environment import SupplyChainEnv
import math


def _run_episode(env: SupplyChainEnv, policy_fn) -> dict:
    """Run a full episode with a given policy function."""
    obs = env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = policy_fn(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    final_state = env.state()
    return {
        "total_reward": total_reward,
        "steps": steps,
        "total_cost": final_state["total_cost"],
        "total_stockouts": final_state["total_stockouts"],
        "total_delivered": final_state["total_delivered"],
    }


def _score_from_episode(result: dict, max_steps: int, node_count: int) -> float:
    """
    Compute a normalised score [0.0, 1.0] from episode stats.
    
    Score = weighted combination of:
      - service_level  (0.4): fraction of demand met without stockout
      - cost_score     (0.3): lower cost → higher score
      - reward_score   (0.3): normalised cumulative reward
    """
    delivered = result["total_delivered"]
    stockouts  = result["total_stockouts"]
    total_demand = delivered + stockouts
    
    service_level = delivered / total_demand if total_demand > 0 else 0.0
    service_level = max(0.0, min(1.0, service_level))
    
    # Cost score: sigmoid inversion (lower cost is better)
    # Scale expected cost by node_count² to better match hard mode reality
    expected_cost = max_steps * node_count * 60.0 * (1 + node_count * 0.1)
    cost_ratio = result["total_cost"] / (expected_cost + 1e-9)
    cost_score = math.exp(-cost_ratio * 0.5)
    cost_score = max(0.0, min(1.0, cost_score))
    
    # Reward score: normalise against a floor/ceiling
    floor   = -(max_steps * node_count * 15.0)
    ceiling =   max_steps * node_count * 1.5
    r = result["total_reward"]
    reward_score = (r - floor) / (ceiling - floor + 1e-9)
    reward_score = max(0.0, min(1.0, reward_score))
    
    final_score = (
        0.4 * service_level +
        0.3 * cost_score    +
        0.3 * reward_score
    )
    return round(final_score, 4)


# ── Policies ─────────────────────────────────────────────────────────────────

def _greedy_policy(obs: dict) -> dict:
    """
    Smart greedy policy: proactively orders, reroutes, and expedites.
    Used as the baseline agent for grading.
    """
    nodes     = obs["nodes"]
    suppliers = obs["suppliers"]
    step      = obs["step"]
    max_steps = obs["max_steps"]

    # Find active suppliers sorted by cost
    active = sorted(
        [s for s in suppliers if s["active"]],
        key=lambda s: s["cost_per_unit"]
    )

    # Critical nodes: inventory < 2 steps of demand
    critical_nodes = [n for n in nodes if n["inventory"] < n["demand_per_step"] * 2]

    # Expedite if critical and active supplier available
    if critical_nodes and active:
        cheapest = active[0]
        qty = min(50, sum(n["demand_per_step"] * 4 for n in critical_nodes))
        return {
            "type": "expedite",
            "supplier_id": cheapest["id"],
            "quantity": max(15, int(qty)),
        }

    # Reroute from overstocked to understocked nodes
    sorted_nodes = sorted(nodes, key=lambda n: n["inventory"])
    if len(sorted_nodes) >= 2:
        poorest = sorted_nodes[0]
        richest = sorted_nodes[-1]
        if (richest["inventory"] > poorest["inventory"] * 4
                and richest["inventory"] > 30
                and poorest["inventory"] < poorest["demand_per_step"] * 5):
            transfer = min(20, richest["inventory"] // 3)
            return {
                "type": "reroute",
                "from_node": richest["id"],
                "to_node":   poorest["id"],
                "transfer_qty": transfer,
            }

    # Proactively restock low nodes before they go critical
    low_nodes = [n for n in nodes
                 if n["inventory"] < n["demand_per_step"] * 5
                 and n["inventory"] < n["capacity"] * 0.5]

    if low_nodes and active:
        cheapest = active[0]
        qty = min(60, sum(n["demand_per_step"] * 6 for n in low_nodes))
        return {
            "type": "order",
            "supplier_id": cheapest["id"],
            "quantity": max(20, int(qty)),
        }

    return {"type": "wait"}


# ── Public grader functions ───────────────────────────────────────────────────

def grade_easy(seed: int = 42) -> float:
    """
    Grade the easy task. Runs a greedy agent and scores it.
    Returns score in [0.0, 1.0].
    """
    env = SupplyChainEnv(difficulty="easy", seed=seed)
    result = _run_episode(env, _greedy_policy)
    score = _score_from_episode(result, max_steps=30, node_count=1)
    print(f"[EASY]   Stockouts={result['total_stockouts']:4d}  "
          f"Cost=${result['total_cost']:7.2f}  "
          f"Delivered={result['total_delivered']:4d}  Score={score:.4f}")
    return score


def grade_medium(seed: int = 42) -> float:
    """
    Grade the medium task. Returns score in [0.0, 1.0].
    """
    env = SupplyChainEnv(difficulty="medium", seed=seed)
    result = _run_episode(env, _greedy_policy)
    score = _score_from_episode(result, max_steps=50, node_count=3)
    print(f"[MEDIUM] Stockouts={result['total_stockouts']:4d}  "
          f"Cost=${result['total_cost']:7.2f}  "
          f"Delivered={result['total_delivered']:4d}  Score={score:.4f}")
    return score


def grade_hard(seed: int = 42) -> float:
    """
    Grade the hard task. Returns score in [0.0, 1.0].
    """
    env = SupplyChainEnv(difficulty="hard", seed=seed)
    result = _run_episode(env, _greedy_policy)
    score = _score_from_episode(result, max_steps=80, node_count=5)
    print(f"[HARD]   Stockouts={result['total_stockouts']:4d}  "
          f"Cost=${result['total_cost']:7.2f}  "
          f"Delivered={result['total_delivered']:4d}  Score={score:.4f}")
    return score


def run_all_graders(seed: int = 42) -> dict:
    """Run all three graders and return scores."""
    print("=" * 55)
    print("  Supply Chain Disruption Manager — Grader Suite")
    print("=" * 55)
    scores = {
        "easy":   grade_easy(seed),
        "medium": grade_medium(seed),
        "hard":   grade_hard(seed),
    }
    print("-" * 55)
    avg = sum(scores.values()) / len(scores)
    print(f"  Average score: {avg:.4f}")
    print("=" * 55)
    return scores


if __name__ == "__main__":
    run_all_graders()
