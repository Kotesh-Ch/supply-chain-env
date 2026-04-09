"""
server/app.py — FastAPI server following OpenEnv spec.
Exposes /reset, /step, /state, /health endpoints on port 7860.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import SupplyChainEnvironment

app = FastAPI(
    title="Supply Chain Disruption Manager",
    description="OpenEnv-compliant RL environment for supply chain management",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session store ─────────────────────────────────────────────────────────────
_envs: dict[str, SupplyChainEnvironment] = {}

def _get_env(session_id: str) -> SupplyChainEnvironment:
    if session_id not in _envs:
        _envs[session_id] = SupplyChainEnvironment()
    return _envs[session_id]


# ── Request / Response schemas ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"
    seed: Optional[int] = None
    session_id: Optional[str] = "default"

class StepRequest(BaseModel):
    action: dict[str, Any]
    session_id: Optional[str] = "default"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment": "supply-chain-disruption-manager",
        "version": "1.0.0",
    }

@app.post("/reset")
def reset(req: ResetRequest):
    env = _get_env(req.session_id)
    obs = env.reset(difficulty=req.difficulty or "easy", seed=req.seed)
    return {"observation": obs, "session_id": req.session_id}

@app.post("/step")
def step(req: StepRequest):
    env = _get_env(req.session_id)
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs,
        "reward":      reward,
        "done":        done,
        "info":        info,
        "session_id":  req.session_id,
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