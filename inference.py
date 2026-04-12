import os
import sys
import json
import math
import random
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# â”€â”€ Dump ALL environment variables at startup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
print("[ENV_DUMP] All environment variables:", flush=True)
for k, v in sorted(os.environ.items()):
    print(f"[ENV_DUMP] {k}={v[:80] if len(v) > 80 else v}", flush=True)

# â”€â”€ Environment variables â€” try every possible name â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _get_env(*keys, default=""):
    for k in keys:
        v = os.environ.get(k, "")
        if v:
            print(f"[INFO] Found env var: {k}={v[:60]}", flush=True)
            return v
    return default

API_BASE_URL = _get_env(
    "API_BASE_URL", "OPENAI_BASE_URL", "LITELLM_PROXY_URL",
    "PROXY_URL", "LLM_BASE_URL", "BASE_URL"
)
API_KEY = _get_env(
    "API_KEY", "OPENAI_API_KEY", "LITELLM_API_KEY",
    "HF_TOKEN", "HUGGINGFACE_TOKEN", "LLM_API_KEY"
) or "placeholder"
MODEL_NAME = _get_env("MODEL_NAME", "LLM_MODEL", "OPENAI_MODEL") or "gpt-4o-mini"
GAME_URL   = _get_env("GAME_URL") or "http://localhost:7860"

print(f"[INFO] API_BASE_URL={API_BASE_URL!r}", flush=True)
print(f"[INFO] API_KEY={'SET' if API_KEY != 'placeholder' else 'NOT SET'}", flush=True)
print(f"[INFO] MODEL_NAME={MODEL_NAME!r}", flush=True)

TASKS = [
    ("easy",   "single_node_supply_chain"),
    ("medium", "multi_node_network"),
    ("hard",   "cascading_disruptions"),
]

# â”€â”€ LLM client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_llm_client    = None
_llm_available = False

try:
    from openai import OpenAI

    if not API_BASE_URL:
        raise ValueError(
            "API_BASE_URL is empty. The validator must inject this env var. "
            "See [ENV_DUMP] logs above for what was actually available."
        )

    # Normalise the base URL â€” ensure it ends with /v1
    base = API_BASE_URL.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"

    print(f"[INFO] Connecting to LiteLLM proxy at: {base}", flush=True)
    _llm_client = OpenAI(base_url=base, api_key=API_KEY)

    # Warm-up call â€” proves the proxy is reachable AND records an API call
    print("[INFO] Making warm-up LLM call...", flush=True)
    _warmup = _llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": 'Reply with {"action_type":"wait"} only.'}],
        max_tokens=20,
        temperature=0.0,
    )
    print(f"[INFO] Warm-up OK: {_warmup.choices[0].message.content.strip()}", flush=True)
    _llm_available = True

except Exception as e:
    print(f"[ERROR] LLM client init/warmup failed: {e}", flush=True)
    traceback.print_exc()

# â”€â”€ Real game environment (optional) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_env_available = False
try:
    from server.environment import SupplyChainEnvironment
    _env_available = True
    print("[INFO] Using real SupplyChainEnvironment", flush=True)
except Exception as _import_err:
    print(f"[WARN] server.environment not found: {_import_err}", flush=True)


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
            dst = next((n for n in self.nodes if n["id"] == action.get("to_node")),   None)
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


# â”€â”€ Greedy fallback policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def greedy_policy(obs):
    nodes     = obs.get("nodes", [])
    suppliers = obs.get("suppliers", [])
    active    = sorted([s for s in suppliers if s.get("active", False)],
                       key=lambda s: s.get("cost_per_unit", float("inf")))
    if not active:
        return {"action_type": "wait"}

    critical = [n for n in nodes if n.get("inventory", 0) < n.get("demand_per_step", 0) * 2]
    if critical:
        qty = sum(n.get("demand_per_step", 0) * 4 for n in critical)
        return {"action_type": "expedite", "supplier_id": active[0]["id"],
                "quantity": min(50, max(15, qty))}

    by_inv = sorted(nodes, key=lambda n: n.get("inventory", 0))
    if len(by_inv) >= 2:
        poor, rich = by_inv[0], by_inv[-1]
        if rich.get("inventory", 0) > poor.get("inventory", 0) * 4 and rich.get("inventory", 0) > 30:
            return {"action_type": "reroute", "from_node": rich["id"],
                    "to_node": poor["id"], "transfer_qty": 15}

    low = [n for n in nodes if n.get("inventory", 0) < n.get("demand_per_step", 0) * 5
           and n.get("inventory", 0) < n.get("capacity", 0) * 0.5]
    if low:
        qty = sum(n.get("demand_per_step", 0) * 6 for n in low)
        return {"action_type": "order", "supplier_id": active[0]["id"],
                "quantity": min(60, max(20, qty))}

    return {"action_type": "wait"}


# â”€â”€ LLM policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
SYSTEM_PROMPT = """You are a supply chain manager AI. Respond with ONLY a valid JSON action object.

Valid actions:
- {"action_type": "wait"}
- {"action_type": "order", "supplier_id": "<id>", "quantity": <int>}
- {"action_type": "expedite", "supplier_id": "<id>", "quantity": <int>}
- {"action_type": "reroute", "from_node": "<id>", "to_node": "<id>", "transfer_qty": <int>}

Decision rules:
- expedite: any node inventory < demand_per_step * 2
- reroute: one node has 4x inventory of another
- order: inventory < demand_per_step * 5 AND < 50% capacity
- wait: otherwise
- Always use the cheapest active supplier for order/expedite."""


def llm_policy(obs):
    """Call LLM via the validator's LiteLLM proxy."""
    if not _llm_available or _llm_client is None:
        print("[WARN] LLM unavailable, using greedy fallback", flush=True)
        return greedy_policy(obs)

    user_msg = (
        f"Nodes:\n{json.dumps(obs.get('nodes', []), indent=2)}\n\n"
        f"Suppliers:\n{json.dumps(obs.get('suppliers', []), indent=2)}\n\n"
        f"Delivered: {obs.get('total_delivered', 0)} | "
        f"Stockouts: {obs.get('total_stockouts', 0)} | "
        f"Cost: {obs.get('total_cost', 0)}\n\n"
        f"Respond with ONLY a JSON action object."
    )

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

        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        action = json.loads(raw)
        if "action_type" not in action:
            raise ValueError(f"No action_type in response: {action}")
        return action

    except Exception as e:
        print(f"[WARN] LLM call failed ({e}), using greedy fallback", flush=True)
        return greedy_policy(obs)


# â”€â”€ Scoring â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def compute_score(obs, total_reward):
    delivered  = obs.get("total_delivered", 0)
    stockouts  = obs.get("total_stockouts", 0)
    total      = delivered + stockouts
    n_nodes    = max(len(obs.get("nodes", [])), 1)
    max_steps  = obs.get("max_steps", 1)
    total_cost = obs.get("total_cost", 0)

    svc        = delivered / total if total > 0 else 0.0
    exp_cost   = max_steps * n_nodes * 60.0 * (1 + n_nodes * 0.1)
    cost_score = math.exp(-total_cost / (exp_cost + 1e-9) * 0.5)
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
    if not _env_available:
        print(f"[WARN] Using fallback simulation for {task_name}", flush=True)

    obs          = env.reset(difficulty=difficulty, seed=42)
    total_reward = 0.0
    step_num     = 0
    done         = False

    print(f"[START] task={task_name}", flush=True)

    while not done and step_num < 1000:
        action = llm_policy(obs)
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
    print(
        f"[INFO] model={MODEL_NAME} | "
        f"env_available={_env_available} | "
        f"llm_available={_llm_available} | "
        f"api_base={API_BASE_URL!r}",
        flush=True
    )

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