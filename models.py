from pydantic import BaseModel
from typing import List, Optional


class Supplier(BaseModel):
    id: str
    name: str
    active: bool
    reliability: float
    lead_time: int
    cost_per_unit: float


class Node(BaseModel):
    id: str
    inventory: int
    capacity: int
    demand_per_step: int
    backlog: int


class Observation(BaseModel):
    step: int
    max_steps: int
    difficulty: str
    suppliers: List[Supplier]
    nodes: List[Node]
    in_transit: List[dict]
    total_cost: float
    total_stockouts: int
    total_delivered: int


class Action(BaseModel):
    type: str
    supplier_id: Optional[str] = None
    quantity: Optional[int] = None
    from_node: Optional[str] = None
    to_node: Optional[str] = None
    transfer_qty: Optional[int] = None


class Reward(BaseModel):
    value: float