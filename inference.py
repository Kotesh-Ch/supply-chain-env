import os
import sys
import json
import math
import random
import traceback
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── ENV VARIABLES ─────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
GAME_URL     = os.environ.get("GAME_URL", "http://localhost:7860")

# normalize /v1
API_BASE_URL_V1 = API_BASE_URL
if API_BASE_URL_V1 and not API_BASE_URL_V1.endswith("/v1"):
    API_BASE_URL_V1 += "/v1"

print(f"[INFO] API_BASE_URL={API_BASE_URL}")
print(f"[INFO] HF_TOKEN={'SET' if HF_TOKEN else 'NOT SET'}")
print(f"[INFO] MODEL_NAME={MODEL_NAME}")

# ── LLM CLIENT ────────────────────────────────
_llm_client = None

try:
    from openai import OpenAI

    if not API_BASE_URL_V1:
        raise ValueError("Missing API_BASE_URL")
    if not HF_TOKEN:
        raise ValueError("Missing HF_TOKEN")

    _llm_client = OpenAI(
        base_url=API_BASE_URL_V1,
        api_key=HF_TOKEN
    )

    # warmup (VERY IMPORTANT for validator)
    print("[INFO] LLM warmup...")
    res = _llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": '{"action_type":"wait"}'}],
        max_tokens=10,
        temperature=0
    )
    print("[INFO] Warmup success:", res.choices[0].message.content)

except Exception as e:
    print("[FATAL] LLM init failed:", e)
    traceback.print_exc()
    sys.exit(1)   # ❌ EXIT if no LLM

# ── GAME CLIENT ───────────────────────────────
class GameClient:
    def __init__(self, base_url=GAME_URL):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty="easy", seed=42):
        r = requests.post(f"{self.base_url}/reset",
                          json={"difficulty": difficulty, "seed": seed})
        r.raise_for_status()
        return r.json()

    def step(self, action):
        r = requests.post(f"{self.base_url}/step",
                          json={"action": action})
        r.raise_for_status()
        return r.json()

# ── LLM POLICY (NO FALLBACK) ──────────────────
SYSTEM_PROMPT = """Return ONLY JSON action.

Allowed:
{"action_type":"wait"}
{"action_type":"order","supplier_id":"S1","quantity":10}
{"action_type":"expedite","supplier_id":"S1","quantity":10}
{"action_type":"reroute","from_node":"n1","to_node":"n2","transfer_qty":10}
"""

def llm_policy(obs):
    response = _llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(obs)}
        ],
        temperature=0,
        max_tokens=100
    )

    raw = response.choices[0].message.content.strip()

    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    action = json.loads(raw.strip())

    if "action_type" not in action:
        raise ValueError("Invalid LLM response")

    return action

# ── SCORE ─────────────────────────────────────
def compute_score(obs, total_reward):
    delivered = obs.get("total_delivered", 0)
    stockouts = obs.get("total_stockouts", 0)
    total = delivered + stockouts

    if total == 0:
        return 0.0

    service = delivered / total
    return round(service, 4)

# ── RUN TASK ──────────────────────────────────
def run_task(client, difficulty, name):
    obs = client.reset(difficulty=difficulty)
    total_reward = 0
    step = 0

    print(f"[START] task={name}", flush=True)

    while True:
        action = llm_policy(obs)
        result = client.step(action)

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        total_reward += reward
        step += 1

        print(f"[STEP] step={step} reward={reward}", flush=True)

        if done:
            break

    score = compute_score(obs, total_reward)

    print(f"[END] task={name} score={score} steps={step}", flush=True)

    return score

# ── MAIN ──────────────────────────────────────
def main():
    client = GameClient()

    tasks = [
        ("easy", "single_node_supply_chain"),
        ("medium", "multi_node_network"),
        ("hard", "cascading_disruptions")
    ]

    scores = []

    for diff, name in tasks:
        try:
            score = run_task(client, diff, name)
            scores.append(score)
        except Exception as e:
            print(f"[ERROR] {name} failed:", e)
            traceback.print_exc()
            scores.append(0)

    avg = sum(scores) / len(scores)
    print(f"[INFO] average_score={round(avg,4)}", flush=True)


if __name__ == "__main__":
    main()