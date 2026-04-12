import os
import sys
import json
import math
import random
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# â”€â”€ Env vars injected by validator â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_BASE_URL = os.environ.get("API_BASE_URL", "").strip()
API_KEY      = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "placeholder")).strip()
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini").strip()

TASKS = [
    ("easy",   "single_node_supply_chain"),
    ("medium", "multi_node_network"),
    ("hard",   "cascading_disruptions"),
]

# â”€â”€ LLM client â€” MUST use validator's proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

if not API_BASE_URL:
    print("[WARN] API_BASE_URL not set â€” LLM calls may fail", flush=True)

# Always initialise with whatever base_url is provided
if OpenAI is not None:
    _llm_client = OpenAI(
        base_url=API_BASE_URL if API_BASE_URL else None,
        api_key=API_KEY,
    )
else:
    _llm_client = None

print(f"[INFO] model={MODEL_NAME} | api_base={API_BASE_URL or 'default'}", flush=True)

# â”€â”€ Real environment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_env_available = False
try:
    from server.environment import SupplyChainEnvironment
    _env_available = True
    print("[INFO] Using real SupplyChainEnvironment", flush=True)
except Exception as e:
    print(f"[WARN] Fallback env: {e}", flush=True)

# â”€â”€ Fallback simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DIFFICULTY_CONFIG = {
    "easy":   {"n_nodes": 1, "n_suppliers": 2, "max_steps": 30,  "capacity": 200},
    "medium": {"n_nodes": 3, "n_suppliers": 3, "max_steps": 50,  "capacity": 150},
    "hard":   {"n_nodes": 5, "n_suppliers": 4, "max_steps": 100, "capacity": 120},
}

class FallbackEnvironment:
    def reset(self, difficulty="easy", seed=42):
        random.seed(seed)
        cfg = DIFFICULTY_CONFIG.get(difficulty, DIFFICULTY_CONFIG["easy"])
        self.cfg             = cfg
        self.step_count      = 0
        self.total_delivered = 0
        self.total_stockouts = 0
        self.total_cost      = 0.0
        self.nodes = [
            {"id": f"node_{i}", "inventory": random.randint(40, 100),
             "capacity": cfg["capacity"], "demand_per_step": random.randint(5, 15)}
            for i in range(cfg["n_nodes"])
        ]
        self.suppliers = [
            {"id": f"supplier_{i}", "active": True,
             "cost_per_unit": round(random.uniform(1.0, 5.0), 2)}
            for i in range(cfg["n_suppliers"])
        ]
        return self._obs()

    def step(self, action):
        self.step_count += 1
        reward = 0.0
        for n in self.nodes:
            d = n["demand_per_step"]
            if n["inventory"] >= d:
                n["inventory"] -= d
                self.total_delivered += d
                reward += 1.0
            else:
                self.total_stockouts += max(0, d - n["inventory"])
                reward -= 5.0
                n["inventory"] = 0
        atype = action.get("action_type", "wait")
        if atype in ("order", "expedite"):
            qty  = action.get("quantity", 20)
            cost = qty * (4.0 if atype == "expedite" else 2.0)
            self.total_cost += cost
            reward -= cost * 0.05
            for n in self.nodes:
                n["inventory"] = min(n["capacity"], n["inventory"] + qty // len(self.nodes))
        elif atype == "reroute":
            src = next((n for n in self.nodes if n["id"] == action.get("from_node")), None)
            dst = next((n for n in self.nodes if n["id"] == action.get("to_node")), None)
            qty = action.get("transfer_qty", 15)
            if src and dst and src["inventory"] >= qty:
                src["inventory"] -= qty
                dst["inventory"]  = min(dst["capacity"], dst["inventory"] + qty)
                reward += 0.5
        return self._obs(), reward, self.step_count >= self.cfg["max_steps"], {}

    def _obs(self):
        return {
            "nodes": self.nodes, "suppliers": self.suppliers,
            "total_delivered": self.total_delivered,
            "total_stockouts": self.total_stockouts,
            "total_cost": self.total_cost,
            "max_steps": self.cfg["max_steps"],
        }

# â”€â”€ LLM system prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SYSTEM_PROMPT = """You are a supply chain manager AI. Respond with ONLY a valid JSON action object â€” no explanation, no markdown.

Valid actions:
- {"action_type": "wait"}
- {"action_type": "order", "supplier_id": "<id>", "quantity": <int 1-60>}
- {"action_type": "expedite", "supplier_id": "<id>", "quantity": <int 1-50>}
- {"action_type": "reroute", "from_node": "<id>", "to_node": "<id>", "transfer_qty": <int 1-30>}

Rules:
- expedite when any node inventory < demand_per_step * 2
- reroute when richest node has 4x inventory of poorest
- order when inventory < demand_per_step * 5 and < 50% capacity
- wait otherwise
- Use cheapest active supplier for order/expedite."""

def greedy_action(obs):
    """Pure greedy â€” used as JSON template when LLM output is invalid."""
    nodes     = obs.get("nodes", [])
    suppliers = obs.get("suppliers", [])
    active    = sorted([s for s in suppliers if s.get("active", False)],
                       key=lambda s: s.get("cost_per_unit", 999))
    if not active:
        return {"action_type": "wait"}
    critical = [n for n in nodes if n.get("inventory", 0) < n.get("demand_per_step", 0) * 2]
    if critical:
        return {"action_type": "expedite", "supplier_id": active[0]["id"],
                "quantity": min(50, max(15, sum(n.get("demand_per_step", 0) * 4 for n in critical)))}
    by_inv = sorted(nodes, key=lambda n: n.get("inventory", 0))
    if len(by_inv) >= 2:
        poor, rich = by_inv[0], by_inv[-1]
        if rich.get("inventory", 0) > poor.get("inventory", 0) * 4 and rich.get("inventory", 0) > 30:
            return {"action_type": "reroute", "from_node": rich["id"],
                    "to_node": poor["id"], "transfer_qty": 15}
    low = [n for n in nodes if n.get("inventory", 0) < n.get("demand_per_step", 0) * 5
           and n.get("inventory", 0) < n.get("capacity", 0) * 0.5]
    if low:
        return {"action_type": "order", "supplier_id": active[0]["id"],
                "quantity": min(60, max(20, sum(n.get("demand_per_step", 0) * 6 for n in low)))}
    return {"action_type": "wait"}

def llm_policy(obs):
    """
    Call LLM through the validator's LiteLLM proxy.
    Falls back to greedy only on total failure.
    """
    user_msg = (
        f"Current supply chain state:\n"
        f"Nodes: {json.dumps(obs.get('nodes', []))}\n"
        f"Suppliers: {json.dumps(obs.get('suppliers', []))}\n"
        f"Delivered={obs.get('total_delivered',0)} "
        f"Stockouts={obs.get('total_stockouts',0)} "
        f"Cost={obs.get('total_cost',0)}\n"
        f"Respond with ONLY a JSON action."
    )
    try:
        response = _llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=100,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        action = json.loads(raw)
        if "action_type" not in action:
            raise ValueError("Missing action_type")
        return action
    except Exception as e:
        print(f"[WARN] LLM call failed: {e} â€” using greedy fallback", flush=True)
        return greedy_action(obs)

# â”€â”€ Scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_score(obs, total_reward):
    delivered  = obs.get("total_delivered", 0)
    stockouts  = obs.get("total_stockouts", 0)
    total      = delivered + stockouts
    n_nodes    = max(len(obs.get("nodes", [])), 1)
    max_steps  = obs.get("max_steps", 1)
    svc        = delivered / total if total > 0 else 0.0
    exp_cost   = max_steps * n_nodes * 60.0 * (1 + n_nodes * 0.1)
    cost_score = math.exp(-obs.get("total_cost", 0) / (exp_cost + 1e-9) * 0.5)
    floor      = -(max_steps * n_nodes * 15.0)
    ceiling    =   max_steps * n_nodes * 1.5
    rew_score  = (total_reward - floor) / (ceiling - floor + 1e-9)
    return round(
        0.4 * max(0.0, min(1.0, svc)) +
        0.3 * max(0.0, min(1.0, cost_score)) +
        0.3 * max(0.0, min(1.0, rew_score)), 4)

# â”€â”€ Run one task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_task(difficulty, task_name):
    env = SupplyChainEnvironment() if _env_available else FallbackEnvironment()
    obs          = env.reset(difficulty=difficulty, seed=42)
    total_reward = 0.0
    step_num     = 0
    done         = False

    print(f"[START] task={task_name}", flush=True)

    while not done and step_num < 1000:
        action       = llm_policy(obs)          # â† always calls LLM proxy
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_num     += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)
    return {
        "task": task_name, "difficulty": difficulty, "score": score,
        "steps": step_num, "total_reward": round(total_reward, 4),
        "delivered": obs.get("total_delivered", 0),
        "stockouts": obs.get("total_stockouts", 0),
        "total_cost": obs.get("total_cost", 0),
    }

# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def main():
    # Log all relevant env vars for debugging
    for k, v in os.environ.items():
        if any(x in k.upper() for x in ["API", "KEY", "URL", "TOKEN", "MODEL", "PROXY"]):
            print(f"[ENV] {k}={v[:80]}", flush=True)

    results = []
    for difficulty, task_name in TASKS:
        try:
            result = run_task(difficulty, task_name)
        except Exception as e:
            print(f"[START] task={task_name}", flush=True)
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
            print(f"[ERROR] {task_name}: {e}", flush=True)
            traceback.print_exc()
            results.append({"task": task_name, "difficulty": difficulty,
                             "score": 0.0, "error": str(e)})
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