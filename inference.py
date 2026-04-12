"""
inference.py — Baseline inference script for OpenEnv hackathon validator.
"""

import os
import sys
import json
import math
import traceback

# Move sys.path setup to module level so the import works from any CWD
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME   = os.getenv("MODEL_NAME",   "supply-chain-greedy-v1")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

TASKS = [
    ("easy",   "single_node_supply_chain"),
    ("medium", "multi_node_network"),
    ("hard",   "cascading_disruptions"),
]

# Try importing the environment once at module level
_env_available = False
try:
    from server.environment import SupplyChainEnvironment
    _env_available = True
except Exception as _import_err:
    print(f"[WARN] Could not import SupplyChainEnvironment: {_import_err}", flush=True)


# ── Policy ────────────────────────────────────────────────────────────────────

def greedy_policy(obs: dict) -> dict:
    # ... (unchanged)


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_score(obs: dict, total_reward: float) -> float:
    # ... (unchanged)


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task_direct(difficulty: str, task_name: str) -> dict:
    if not _env_available:
        raise RuntimeError("SupplyChainEnvironment not importable; cannot use direct mode.")
    
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
        "task": task_name, "difficulty": difficulty, "score": score,
        "steps": step_num, "total_reward": round(total_reward, 4),
        "delivered": obs["total_delivered"], "stockouts": obs["total_stockouts"],
        "total_cost": obs["total_cost"],
    }


def run_task_http(difficulty: str, task_name: str) -> dict:
    import requests
    base = API_BASE_URL.rstrip("/")

    r = requests.post(f"{base}/reset", json={"difficulty": difficulty, "seed": 42}, timeout=30)
    r.raise_for_status()
    obs = r.json()["observation"]
    total_reward = 0.0
    step_num = 0
    done = False

    print(f"[START] task={task_name}", flush=True)
    while not done:
        action = greedy_policy(obs)
        r = requests.post(f"{base}/step", json={"action": action}, timeout=30)
        r.raise_for_status()
        data = r.json()
        obs, reward, done = data["observation"], data["reward"], data["done"]
        total_reward += reward
        step_num += 1
        print(f"[STEP] step={step_num} reward={round(reward, 4)}", flush=True)

    score = compute_score(obs, total_reward)
    print(f"[END] task={task_name} score={score} steps={step_num}", flush=True)
    return {
        "task": task_name, "difficulty": difficulty, "score": score,
        "steps": step_num, "total_reward": round(total_reward, 4),
        "delivered": obs["total_delivered"], "stockouts": obs["total_stockouts"],
        "total_cost": obs["total_cost"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    use_http = (
        API_BASE_URL != "http://localhost:7860"
        and not API_BASE_URL.startswith("http://localhost")
        and not API_BASE_URL.startswith("http://127.")
    )

    print(f"[INFO] Supply Chain Disruption Manager | model={MODEL_NAME} | "
          f"mode={'http' if use_http else 'direct'} | env_available={_env_available}", flush=True)

    results = []
    failed_tasks = []

    for difficulty, task_name in TASKS:
        try:
            if use_http:
                result = run_task_http(difficulty, task_name)
            else:
                result = run_task_direct(difficulty, task_name)
        except Exception as e:
            print(f"[WARN] Primary mode failed ({e}), trying fallback...", flush=True)
            try:
                if use_http and _env_available:
                    result = run_task_direct(difficulty, task_name)
                elif not use_http:
                    result = run_task_http(difficulty, task_name)
                else:
                    raise RuntimeError("No fallback mode available.")
            except Exception as e2:
                print(f"[ERROR] Both modes failed for {task_name}: {e2}", flush=True)
                results.append({"task": task_name, "difficulty": difficulty, "score": 0.0,
                                 "error": str(e2)})
                failed_tasks.append(task_name)
                continue

        results.append(result)
        if result["score"] < 0.4:
            failed_tasks.append(task_name)

    print("[INFO] === FINAL RESULTS ===", flush=True)
    print(json.dumps(results, indent=2), flush=True)

    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"[INFO] average_score={round(avg, 4)}", flush=True)

    sys.exit(0 if not failed_tasks else 1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)   # Always exit cleanly, never let an unhandled exception propagate