"""
Supply Chain Disruption Manager - OpenEnv Environment
"""

import random
from typing import Optional
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

@dataclass
class Supplier:
    id: str
    name: str
    active: bool
    reliability: float
    lead_time: int
    cost_per_unit: float


@dataclass
class Node:
    id: str
    inventory: int
    capacity: int
    demand_per_step: int
    backlog: int = 0


@dataclass
class Shipment:
    supplier_id: str
    quantity: int
    eta: int


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

class SupplyChainEnv:

    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        assert difficulty in self.DIFFICULTIES
        self.difficulty = difficulty
        self.seed = seed
        self._rng = random.Random(seed)
        self._max_steps = {"easy": 30, "medium": 50, "hard": 80}[difficulty]
        self._setup_scenario()

    # ─────────────────────────────────────────────

    def _setup_scenario(self):
        if self.difficulty == "easy":
            self.suppliers = [
                Supplier("S1", "AlphaSupply", True, 0.95, 2, 5.0),
            ]
            self.nodes = [
                Node("W1", 50, 100, 8),
            ]

        elif self.difficulty == "medium":
            self.suppliers = [
                Supplier("S1", "AlphaSupply", True, 0.85, 2, 5.0),
                Supplier("S2", "BetaLogistics", True, 0.75, 3, 4.0),
            ]
            self.nodes = [
                Node("W1", 60, 120, 10),
                Node("W2", 40, 100, 8),
                Node("W3", 30, 80, 6),
            ]

        else:
            self.suppliers = [
                Supplier("S1", "AlphaSupply", True, 0.80, 2, 5.0),
                Supplier("S2", "BetaLogistics", True, 0.70, 3, 4.0),
                Supplier("S3", "GammaFreight", True, 0.65, 1, 7.0),
                Supplier("S4", "DeltaExpress", True, 0.90, 4, 8.0),
            ]
            self.nodes = [
                Node("W1", 80, 200, 8),
                Node("W2", 60, 160, 7),
                Node("W3", 40, 140, 6),
                Node("W4", 30, 120, 5),
                Node("W5", 20, 100, 4),
            ]

        self.in_transit = []
        self._step_count = 0
        self._total_cost = 0.0
        self._total_stockouts = 0
        self._total_delivered = 0

    # ─────────────────────────────────────────────

    def _encode_state(self):
        return {
            "step": self._step_count,
            "max_steps": self._max_steps,
            "difficulty": self.difficulty,
            "suppliers": [asdict(s) for s in self.suppliers],
            "nodes": [asdict(n) for n in self.nodes],
            "in_transit": [asdict(s) for s in self.in_transit],
            "total_cost": round(self._total_cost, 2),
            "total_stockouts": self._total_stockouts,
            "total_delivered": self._total_delivered,
        }

    # ─────────────────────────────────────────────

    def _apply_disruptions(self):
        if self.difficulty == "easy":
            return

        for s in self.suppliers:
            roll = self._rng.random()
            if s.active and roll > s.reliability:
                s.active = False
            elif not s.active and roll < 0.4:
                s.active = True

    # ─────────────────────────────────────────────

    def _advance_shipments(self):
        arrived = []
        remaining = []

        for ship in self.in_transit:
            ship.eta -= 1
            if ship.eta <= 0:
                arrived.append(ship)
            else:
                remaining.append(ship)

        self.in_transit = remaining

        for ship in arrived:
            qty = ship.quantity
            needed = sorted(self.nodes, key=lambda n: n.inventory)

            for node in needed:
                space = node.capacity - node.inventory
                add = min(space, qty)

                node.inventory += add
                self._total_delivered += add

                qty -= add
                if qty <= 0:
                    break

    # ─────────────────────────────────────────────

    def _consume_demand(self):
        penalty = 0.0

        for node in self.nodes:
            demand = node.demand_per_step + node.backlog

            if node.inventory >= demand:
                node.inventory -= demand
                node.backlog = 0
            else:
                shortfall = demand - node.inventory
                node.inventory = 0
                node.backlog = shortfall

                penalty += shortfall * 10
                self._total_stockouts += shortfall

        return penalty

    # ─────────────────────────────────────────────

    def reset(self):
        self._rng = random.Random(self.seed)
        self._setup_scenario()
        return self._encode_state()

    # ─────────────────────────────────────────────

    def step(self, action: dict):
        self._step_count += 1
        reward = 0.0

        info = {"action_taken": "unknown", "events": []}

        # disruptions
        self._apply_disruptions()

        info["events"].append(
            f"Suppliers status: {[(s.id, s.active) for s in self.suppliers]}"
        )

        action_type = action.get("type", "wait")

        # ORDER
        if action_type == "order":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity", 0))
            supplier = next((s for s in self.suppliers if s.id == sid), None)

            if not supplier:
                reward -= 5
                info["action_taken"] = "order_failed"
                info["events"].append("Invalid supplier")

            elif not supplier.active:
                reward -= 5
                info["action_taken"] = "order_failed"
                info["events"].append("Supplier inactive")

            elif qty <= 0:
                reward -= 2
                info["action_taken"] = "order_failed"

            else:
                cost = supplier.cost_per_unit * qty
                self._total_cost += cost
                reward -= cost * 0.1

                self.in_transit.append(Shipment(sid, qty, supplier.lead_time))

                info["action_taken"] = "order_success"

        # EXPEDITE
        elif action_type == "expedite":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity", 0))
            supplier = next((s for s in self.suppliers if s.id == sid), None)

            if supplier and supplier.active and qty > 0:
                cost = supplier.cost_per_unit * qty * 1.8
                self._total_cost += cost
                reward -= cost * 0.1

                self.in_transit.append(Shipment(sid, qty, 1))
                info["action_taken"] = "expedite_success"
            else:
                reward -= 5
                info["action_taken"] = "expedite_failed"

        # WAIT
        else:
            info["action_taken"] = "wait"

        # progress
        self._advance_shipments()
        reward -= self._consume_demand()

        for node in self.nodes:
            ratio = node.inventory / node.capacity
            if 0.2 <= ratio <= 0.8:
                reward += 1

        done = self._step_count >= self._max_steps

        return self._encode_state(), round(reward, 4), done, info

    # ─────────────────────────────────────────────

    def state(self):
        return self._encode_state()