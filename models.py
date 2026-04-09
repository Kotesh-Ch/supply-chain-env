"""
models.py — Typed Pydantic models for Supply Chain Disruption Manager.
Defines Action, Observation, and State used by server and client.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal


# ── Action ────────────────────────────────────────────────────────────────────

class SupplyChainAction(BaseModel):
    action_type: Literal["order", "expedite", "reroute", "wait"] = Field(
        default="wait",
        description="Type of supply chain action to take"
    )
    supplier_id: Optional[str] = Field(
        default=None,
        description="Supplier ID for order/expedite actions (e.g. 'S1')"
    )
    quantity: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description="Units to order or expedite"
    )
    from_node: Optional[str] = Field(
        default=None,
        description="Source warehouse node for reroute (e.g. 'W1')"
    )
    to_node: Optional[str] = Field(
        default=None,
        description="Destination warehouse node for reroute (e.g. 'W2')"
    )
    transfer_qty: Optional[int] = Field(
        default=None,
        ge=1, le=100,
        description="Units to transfer between nodes"
    )


# ── Sub-models ────────────────────────────────────────────────────────────────

class SupplierInfo(BaseModel):
    id: str
    name: str
    active: bool
    reliability: float
    lead_time: int
    cost_per_unit: float

class NodeInfo(BaseModel):
    id: str
    inventory: int
    capacity: int
    demand_per_step: int
    backlog: int = 0

class ShipmentInfo(BaseModel):
    supplier_id: str
    quantity: int
    eta: int


# ── Observation ───────────────────────────────────────────────────────────────

class SupplyChainObservation(BaseModel):
    step: int
    max_steps: int
    difficulty: str
    suppliers: List[SupplierInfo]
    nodes: List[NodeInfo]
    in_transit: List[ShipmentInfo]
    total_cost: float
    total_stockouts: int
    total_delivered: int
    reward: float = 0.0
    done: bool = False
    success: bool = True
    message: str = ""


# ── State ─────────────────────────────────────────────────────────────────────

class SupplyChainState(BaseModel):
    step_count: int = 0
    difficulty: str = "easy"
    total_cost: float = 0.0
    total_stockouts: int = 0
    total_delivered: int = 0
    episode_reward: float = 0.0