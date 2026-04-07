"""
validate.py — Pre-submission validation script.
Run this before submitting to catch disqualifying issues early.

Usage: python validate.py
"""

import sys
import json
import importlib
import traceback


PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(name: str, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
        return True
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name}")
        print(f"       → {e}")
        return False


print("\n" + "=" * 60)
print("  Supply Chain Disruption Manager — Pre-Submission Validator")
print("=" * 60 + "\n")


# 1. Import checks
print("[ IMPORTS ]")

def check_environment():
    import environment
    assert hasattr(environment, "SupplyChainEnv")

def check_server():
    import server
    assert hasattr(server, "app")

def check_graders():
    import graders
    assert hasattr(graders, "grade_easy")
    assert hasattr(graders, "grade_medium")
    assert hasattr(graders, "grade_hard")

def check_inference():
    import inference
    assert hasattr(inference, "greedy_policy")

check("environment.py imports OK",  check_environment)
check("server.py imports OK",       check_server)
check("graders.py imports OK",      check_graders)
check("inference.py imports OK",    check_inference)


# 2. OpenEnv API compliance
print("\n[ OPENENV API ]")

from environment import SupplyChainEnv

def check_reset():
    env = SupplyChainEnv()
    obs = env.reset()
    assert isinstance(obs, dict), "reset() must return dict"
    assert "step" in obs
    assert "nodes" in obs
    assert "suppliers" in obs

def check_step():
    env = SupplyChainEnv()
    env.reset()
    result = env.step({"type": "wait"})
    assert isinstance(result, tuple) and len(result) == 4, \
        "step() must return (obs, reward, done, info)"
    obs, reward, done, info = result
    assert isinstance(obs, dict)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def check_state():
    env = SupplyChainEnv()
    env.reset()
    s = env.state()
    assert isinstance(s, dict)

def check_full_episode():
    env = SupplyChainEnv(difficulty="easy", seed=0)
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        _, _, done, _ = env.step({"type": "wait"})
        steps += 1
    assert steps == 30, f"Easy episode should be 30 steps, got {steps}"

check("reset() returns dict with required keys", check_reset)
check("step() returns (obs, reward, done, info)", check_step)
check("state() returns dict", check_state)
check("Full easy episode completes in 30 steps", check_full_episode)


# 3. Task graders
print("\n[ GRADERS ]")

from graders import grade_easy, grade_medium, grade_hard

def check_grade_easy():
    score = grade_easy(seed=0)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

def check_grade_medium():
    score = grade_medium(seed=0)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

def check_grade_hard():
    score = grade_hard(seed=0)
    assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

check("grade_easy()   returns score in [0, 1]", check_grade_easy)
check("grade_medium() returns score in [0, 1]", check_grade_medium)
check("grade_hard()   returns score in [0, 1]", check_grade_hard)


# 4. YAML spec
print("\n[ OPENENV YAML ]")

def check_yaml():
    import yaml, pathlib
    path = pathlib.Path("openenv.yaml")
    assert path.exists(), "openenv.yaml not found"
    with open(path) as f:
        data = yaml.safe_load(f)
    assert "tasks" in data, "openenv.yaml must have 'tasks'"
    assert len(data["tasks"]) >= 3, "Need at least 3 tasks"
    for task in data["tasks"]:
        assert "id" in task
        assert "difficulty" in task

check("openenv.yaml exists and has 3+ tasks", check_yaml)


# 5. Dockerfile
print("\n[ DOCKERFILE ]")

def check_dockerfile():
    import pathlib
    path = pathlib.Path("Dockerfile")
    assert path.exists(), "Dockerfile not found"
    content = path.read_text()
    assert "7860" in content, "Dockerfile must EXPOSE 7860"
    assert "HEALTHCHECK" in content, "Dockerfile must have HEALTHCHECK"

check("Dockerfile exists with port 7860 and HEALTHCHECK", check_dockerfile)


# 6. Inference script
print("\n[ INFERENCE SCRIPT ]")

def check_inference_script():
    import pathlib
    assert pathlib.Path("inference.py").exists(), "inference.py must be in root directory"

check("inference.py exists in root directory", check_inference_script)


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  Results: {passed} passed, {failed} failed")

if failed == 0:
    print("  🎉 All checks passed! Safe to submit.")
else:
    print("  ⛔ Fix failing checks before submitting.")
print("=" * 60 + "\n")

sys.exit(0 if failed == 0 else 1)
