"""
server.py — FastAPI server exposing OpenEnv-compliant REST endpoints.
Run with: uv run server  OR  uvicorn server:app --host 0.0.0.0 --port 7860
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn

from environment import SupplyChainEnv

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

# ── In-memory session store ──────────────────────────────────────────────────
_sessions: dict[str, SupplyChainEnv] = {}
_default_session = "default"
_sessions[_default_session] = SupplyChainEnv(difficulty="easy")


# ── Request schemas ──────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"
    seed: Optional[int] = None
    session_id: Optional[str] = _default_session

class StepRequest(BaseModel):
    action: dict[str, Any]
    session_id: Optional[str] = _default_session

class SessionRequest(BaseModel):
    session_id: Optional[str] = _default_session


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "env": "supply-chain-disruption-manager", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest):
    """Reset environment and return initial observation."""
    difficulty = req.difficulty or "easy"
    if difficulty not in SupplyChainEnv.DIFFICULTIES:
        raise HTTPException(400, f"difficulty must be one of {SupplyChainEnv.DIFFICULTIES}")
    env = SupplyChainEnv(difficulty=difficulty, seed=req.seed)
    _sessions[req.session_id] = env
    obs = env.reset()
    return {"observation": obs, "session_id": req.session_id}


@app.post("/step")
def step(req: StepRequest):
    """Take one step in the environment."""
    env = _sessions.get(req.session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /reset first.")
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info,
        "session_id": req.session_id,
    }


@app.get("/state")
def state(session_id: str = _default_session):
    """Return current state without advancing the environment."""
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found. Call /reset first.")
    return {"state": env.state(), "session_id": session_id}


@app.get("/action_space")
def action_space(session_id: str = _default_session):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return env.action_space()


@app.get("/observation_space")
def observation_space(session_id: str = _default_session):
    env = _sessions.get(session_id)
    if env is None:
        raise HTTPException(404, "Session not found.")
    return env.observation_space()


@app.get("/tasks")
def list_tasks():
    """List available tasks/difficulties."""
    return {
        "tasks": [
            {"id": "easy",   "name": "Single-Node Supply Chain",  "max_steps": 30},
            {"id": "medium", "name": "Multi-Node Network",         "max_steps": 50},
            {"id": "hard",   "name": "Cascading Disruptions",      "max_steps": 80},
        ]
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
