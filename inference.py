import os
import sys
import json
import math
import random
import traceback

# Ensure proper import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Environment variables (injected by validator) ─────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
API_KEY      = os.environ.get("API_KEY", "placeholder-key")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

TASKS = [
    ("easy",   "single_node_supply_chain"),
    ("medium", "multi_node_network"),
    ("hard",   "cascading_disruptions"),
]

# ── OpenAI client via LiteLLM proxy ──────────────────────────────────
try:
    from openai import OpenAI
    _llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )
    _llm_available = True
except Exception as e:
    print(f"[WARN] Could not init OpenAI client: {e}", flush=True)
    _llm_client    = None
    _llm_available = False

# ── Real environment (optional) ───────────────────────────────────────
_env_available = False
try:
    from server.environment import SupplyChainEnvironment
    _env_available = True
except Exception as _import_err:
    print(f"[WARN] Could not import SupplyChainEnvironment: {_import_err}", flush=True)


# ── Fallback simulation ───────────────────────────────────────────────

DIFFICULTY_CONFIG = {
    "easy":   {"n_nodes": 1, "n_suppliers": 2, "max_steps": 30,  "capacity": 200},
    "medium": {"n_nodes": 3, "n_suppliers": 3, "max_steps": 50,  "capacity": 150},
    "hard":   {"n_nodes": 5, "n_suppliers": 4, "max_steps": 100, "capacity": 120},
}

class FallbackEnvironment:
    def reset(self, difficulty: str = "easy", seed: int = 42) -> dict:
        random.seed(seed)
        cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["easy"])
        self.cfg             = cfg
        self.step_count      = 0
        self.total_delivered = 0
        self.total_stockouts = 0
        self.total_cost      = 0.0

        self.nodes = [
            {
                "id":              f"node_{i}",
                "inventory":       random.randint(40, 100),
                "capacity":        cfg["capacity"],
                "demand_per_step": random.randint(5, 15),
            }
            for i in range(cfg["n_nodes"])
        ]
        self.suppliers = [
            {
                "id":            f"supplier_{i}",
                "active":        True,
                "cost_per_unit": round(random.uniform(1.0, 5.0), 2),
            }
            for i in range(cfg["n_suppliers"])
        ]
        return self._obs()

    def step(self, action: dict):
        self.step_count += 1
        action_type = action.get("action_type", "wait")
        reward = 0.0

        for n in self.nodes:
            demand = n["demand_per_step"]
            if n["inventory"] >= demand:
                n["inventory"] -= demand
                self.total_delivered += demand
                reward += 1.0
            else:
                self.total_stockouts += max(0, demand - n["inventory"])
                reward -= 5.0
                n["inventory"] = 0

        if action_type in ("order", "expedite"):
            qty  = action.get("quantity", 20)
            cost = qty * (4.0 if action_type == "expedite" else 2.0)
            self.total_cost += cost
            reward -= cost * 0.05
            for n in self.nodes:
                n["inventory"] = min(n["capacity"], n["inventory"] + qty // len(self.nodes))

        elif action_type == "reroute":
            from_id = action.get("from_node")
            to_id   = action.get("to_node")
            qty     = action.get("transfer_qty", 15)
            src = next((n for n in self.nodes if n["id"] == from_id), None)
            dst = next((n for n in self.nodes if n["id"] == to_id),   None)
            if src and dst and src["inventory"] >= qty:
                src["inventory"] -= qty
                dst["inventory"]  = min(dst["capacity"], dst["inventory"] + qty)
                reward += 0.5

        done = self.step_count >= self.cfg["max_steps"]
        return self._obs(), reward, done, {}

    def _obs(self) -> dict:
        return {
            "nodes":           self.nodes,
            "suppliers":       self.suppliers,
            "total_delivered": self.total_delivered,
            "total_stockouts": self.total_stockouts,
            "total_cost":      self.total_cost,
            "max_steps":       self.cfg["max_steps"],
        }


# ── LLM policy ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a supply chain manager AI. Given the current state of nodes and suppliers,
decide the best action. Respond with ONLY a valid JSON object, no explanation.

Valid action types and their required fields:
- {"action_type": "wait"}
- {"action_type": "order", "supplier_id": "<id>", "quantity": <int>}
- {"action_type": "expedite", "supplier_id": "<id>", "quantity": <int>}
- {"action_type": "reroute", "from_node": "<id>", "to_node": "<id>", "transfer_qty": <int>}

Rules:
- Use "expedite" only when any node has inventory < demand_per_step * 2 (critical stockout risk)
- Use "reroute" when one node has 4x more inventory than another
- Use "order" when inventory < demand_per_step * 5 and < 50% capacity
- Otherwise "wait"
- Always pick the cheapest active supplier for order/expedite
"""

def greedy_fallback(obs: dict) -> dict:
    """Pure greedy policy used as fallback if LLM call fails."""
    nodes     = obs.get("nodes", [])
    suppliers = obs.get("suppliers", [])

    active = sorted(
        [s for s in suppliers if s.get("active", False)],
        key=lambda s: s.get("cost_per_unit", float("inf"))
    )
    if not active:
        return {"action_type": "wait"}

    critical = [n for n in nodes if n.get("inventory", 0) < n.get("demand_per_step", 0) * 2]
    if critical:
        qty = sum(n.get("demand_per_step", 0) * 4 for n in critical)
        return {"action_type": "expedite", "supplier_id": active[0].get("id"),
                "quantity": min(50, max(15, qty))}

    by_inv = sorted(nodes, key=lambda n: n.get("inventory", 0))
    if len(by_inv) >= 2:
        poor, rich = by_inv[0], by_inv[-1]
        if rich.get("inventory", 0) > poor.get("inventory", 0) * 4 and rich.get("inventory", 0) > 30:
            return {"action_type": "reroute", "from_node": rich.get("id"),
                    "to_node": poor.get("id"), "transfer_qty": 15}

    low = [n for n in nodes
           if n.get("inventory", 0) < n.get("demand_per_step", 0) * 5
           and n.get("inventory", 0) < n.get("capacity", 0) * 0.5]
    if low:
        qty = sum(n.get("demand_per_step", 0) * 6 for n in low)
        return {"action_type": "order", "supplier_id": active[0].get("id"),
                "quantity": min(60, max(20, qty))}

    return {"action_type": "wait"}


def llm_policy(obs: dict) -> dict:
    """Call LLM through the validator's proxy to decide the action."""
    if not _llm_available or _llm_client is None:
        return greedy_fallback(obs)

    user_msg = f"""Current supply chain state:

Nodes:
{json.dumps(obs.get('nodes', []), indent=2)}

Suppliers:
{json.dumps(obs.get('suppliers', []), indent=2)}

Stats so far:
- Total delivered: {obs.get('total_delivered', 0)}
- Total stockouts: {obs.get('total_stockouts', 0)}
- Total cost: {obs.get('total_cost', 0)}

What action should be taken? Respond with ONLY a JSON object."""

    try:
        response = _llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action = json.loads(raw)

        # Validate action has required field
        if "action_type" not in action:
            raise ValueError(f"Missing action_type in: {action}")

        return action

    except Exception as e:
        print(f"[WARN] LLM call failed ({e}), using greedy fallback", flush=True)
        return greedy_fallback(obs)


# ── Scoring ───────────────────────────────────────────────────────────

def compute_score(obs: dict, total_reward: float) -> float:
    delivered  = obs.get("total_delivered", 0)
    stockouts  = obs.get("total_stockouts", 0)
    total      = delivered + stockouts

    nodes      = obs.get("nodes", [])
    n_nodes    = max(len(nodes), 1)
    max_steps  = obs.get("max_steps", 1)
    total_cost = obs.get("total_cost", 0)

    svc = delivered / total if total > 0 else 0.0

    exp_cost   = max_steps * n_nodes * 60.0 * (1 + n_nodes * 0.1)
    cost_score = math.exp(-total_cost / (exp_cost + 1e-9) * 0.5)

    floor     = -(max_steps * n_nodes * 15.0)
    ceiling   =   max_steps * n_nodes * 1.5
    rew_score = (total_reward - floor) / (ceiling - floor + 1e-9)

    return round(
        0.4 * max(0.0, min(1.0, svc)) +
        0.3 * max(0.0, min(1.0, cost_score)) +
        0.3 * max(0.0, min(1.0, rew_score)),
        4
    )


# ── Run task (direct) ─────────────────────────────────────────────────

def run_task_direct(difficulty: str, task_name: str) -> dict:
    env = SupplyChainEnvironment() if _env_available else FallbackEnvironment()
    if not _env_available:
        print(f"[WARN] Using fallback simulation for {task_name}", flush=True)

    obs            = env.reset(difficulty=difficulty, seed=42)
    total_reward   = 0.0
    step_num       = 0
    done           = False
    max_safe_steps = 1000

    print(f"[START] task={task_name}", flush=True)

    while not done and step_num < max_safe_steps:
        action = llm_policy(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_num     += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)

    return {
        "task":         task_name,
        "difficulty":   difficulty,
        "score":        score,
        "steps":        step_num,
        "total_reward": round(total_reward, 4),
        "delivered":    obs.get("total_delivered", 0),
        "stockouts":    obs.get("total_stockouts", 0),
        "total_cost":   obs.get("total_cost", 0),
    }


# ── Run task (HTTP) ───────────────────────────────────────────────────

def run_task_http(difficulty: str, task_name: str) -> dict:
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed")

    base = API_BASE_URL.rstrip("/")

    r = requests.post(f"{base}/reset",
                      json={"difficulty": difficulty, "seed": 42},
                      timeout=30)
    r.raise_for_status()
    obs = r.json().get("observation", {})

    total_reward   = 0.0
    step_num       = 0
    done           = False
    max_safe_steps = 1000

    print(f"[START] task={task_name}", flush=True)

    while not done and step_num < max_safe_steps:
        action = llm_policy(obs)
        r = requests.post(f"{base}/step",
                          json={"action": action},
                          timeout=30)
        r.raise_for_status()
        data   = r.json()
        obs    = data.get("observation", {})
        reward = data.get("reward", 0)
        done   = data.get("done", False)
        total_reward += reward
        step_num     += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)

    return {
        "task":         task_name,
        "difficulty":   difficulty,
        "score":        score,
        "steps":        step_num,
        "total_reward": round(total_reward, 4),
        "delivered":    obs.get("total_delivered", 0),
        "stockouts":    obs.get("total_stockouts", 0),
        "total_cost":   obs.get("total_cost", 0),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    use_http = (
        API_BASE_URL != "http://localhost:7860"
        and not API_BASE_URL.startswith("http://localhost")
        and not API_BASE_URL.startswith("http://127.")
    )

    print(
        f"[INFO] model={MODEL_NAME} | "
        f"mode={'http' if use_http else 'direct'} | "
        f"env_available={_env_available} | "
        f"llm_available={_llm_available}",
        flush=True
    )

    results = []

    for difficulty, task_name in TASKS:
        try:
            if use_http:
                result = run_task_http(difficulty, task_name)
            else:
                result = run_task_direct(difficulty, task_name)
        except Exception as e:
            print(f"[START] task={task_name}", flush=True)
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
            print(f"[ERROR] Failed task {task_name}: {e}", flush=True)
            traceback.print_exc()
            results.append({
                "task":       task_name,
                "difficulty": difficulty,
                "score":      0.0,
                "error":      str(e),
            })
            continue

        results.append(result)

    print("[INFO] === FINAL RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"[INFO] average_score={round(avg, 4)}", flush=True)

    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(0)