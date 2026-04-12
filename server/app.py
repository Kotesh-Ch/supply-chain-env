"""
server/app.py â€” FastAPI server following OpenEnv spec.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.environment import SupplyChainEnvironment

app = FastAPI(
    title="Supply Chain Disruption Manager",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# â”€â”€ Session store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_envs: dict[str, SupplyChainEnvironment] = {}

def _get_env(session_id: str = "default") -> SupplyChainEnvironment:
    if session_id not in _envs:
        _envs[session_id] = SupplyChainEnvironment()
    return _envs[session_id]


# â”€â”€ Endpoints â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "supply-chain-disruption-manager",
        "version": "1.0.0",
    }

@app.post("/reset")
async def reset(request: Request):
    """Reset endpoint â€” accepts empty body OR JSON body."""
    # Parse body safely â€” empty body is fine
    try:
        body = await request.json()
    except Exception:
        body = {}

    if body is None:
        body = {}

    difficulty = body.get("difficulty", "easy") or "easy"
    seed       = body.get("seed", None)
    session_id = body.get("session_id", "default") or "default"

    env = _get_env(session_id)
    obs = env.reset(difficulty=difficulty, seed=seed)
    return {"observation": obs, "session_id": session_id}

@app.post("/step")
async def step(request: Request):
    """Step endpoint â€” accepts action dict."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    if body is None:
        body = {}

    action     = body.get("action", {"action_type": "wait"}) or {"action_type": "wait"}
    session_id = body.get("session_id", "default") or "default"

    env = _get_env(session_id)
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "session_id":  session_id,
    }

@app.get("/state")
def state(session_id: str = "default"):
    env = _get_env(session_id)
    return {"state": env.state(), "session_id": session_id}

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "easy",   "name": "Single-Node Supply Chain",  "max_steps": 30},
        {"id": "medium", "name": "Multi-Node Network",         "max_steps": 50},
        {"id": "hard",   "name": "Cascading Disruptions",      "max_steps": 80},
    ]}

@app.get("/")
def root():
    return {
        "name": "Supply Chain Disruption Manager",
        "version": "1.0.0",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/docs"],
    }

if __name__ == "__main__":
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)