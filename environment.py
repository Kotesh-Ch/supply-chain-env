"""
Supply Chain Disruption Manager - OpenEnv Environment
A real-world RL environment where an AI agent manages a multi-node supply chain,
handling disruptions, rerouting shipments, and minimizing costs.
"""

import random
import json
from typing import Any, Optional
from dataclasses import dataclass, asdict


# ─────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────

@dataclass
class Supplier:
    id: str
    name: str
    active: bool
    reliability: float       # 0.0–1.0 probability of being operational
    lead_time: int           # days to deliver
    cost_per_unit: float

@dataclass
class Node:
    id: str
    inventory: int
    capacity: int
    demand_per_step: int
    backlog: int = 0         # unfulfilled demand

@dataclass
class Shipment:
    supplier_id: str
    quantity: int
    eta: int                 # steps until arrival


# ─────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────

class SupplyChainEnv:
    """
    OpenEnv-compliant Supply Chain Disruption Manager.

    Difficulty levels:
      easy   — 1 supplier, 1 node, no disruptions
      medium — 2 suppliers, 3 nodes, random disruptions
      hard   — 4 suppliers, 5 nodes, cascading disruptions + demand spikes
    """

    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self, difficulty: str = "easy", seed: Optional[int] = None):
        assert difficulty in self.DIFFICULTIES, f"difficulty must be one of {self.DIFFICULTIES}"
        self.difficulty = difficulty
        self.seed = seed
        self._rng = random.Random(seed)
        self._step_count = 0
        self._max_steps = {"easy": 30, "medium": 50, "hard": 80}[difficulty]
        self._setup_scenario()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _setup_scenario(self):
        if self.difficulty == "easy":
            self.suppliers = [
                Supplier("S1", "AlphaSupply", True, 0.95, 2, 5.0),
            ]
            self.nodes = [
                Node("W1", inventory=50, capacity=100, demand_per_step=8),
            ]
        elif self.difficulty == "medium":
            self.suppliers = [
                Supplier("S1", "AlphaSupply",  True, 0.85, 2, 5.0),
                Supplier("S2", "BetaLogistics", True, 0.75, 3, 4.0),
            ]
            self.nodes = [
                Node("W1", inventory=60, capacity=120, demand_per_step=10),
                Node("W2", inventory=40, capacity=100, demand_per_step=8),
                Node("W3", inventory=30, capacity=80,  demand_per_step=6),
            ]
        else:  # hard
            self.suppliers = [
                Supplier("S1", "AlphaSupply",    True, 0.80, 2, 5.0),
                Supplier("S2", "BetaLogistics",  True, 0.70, 3, 4.0),
                Supplier("S3", "GammaFreight",   True, 0.65, 1, 7.0),
                Supplier("S4", "DeltaExpress",   True, 0.90, 4, 8.0),
            ]
            self.nodes = [
                Node("W1", inventory=80,  capacity=200, demand_per_step=8),
                Node("W2", inventory=60,  capacity=160, demand_per_step=7),
                Node("W3", inventory=40,  capacity=140, demand_per_step=6),
                Node("W4", inventory=30,  capacity=120, demand_per_step=5),
                Node("W5", inventory=20,  capacity=100, demand_per_step=4),
            ]

        self.in_transit: list[Shipment] = []
        self._step_count = 0
        self._total_cost = 0.0
        self._total_stockouts = 0
        self._total_delivered = 0

    def _encode_state(self) -> dict:
        return {
            "step": self._step_count,
            "max_steps": self._max_steps,
            "difficulty": self.difficulty,
            "suppliers": [asdict(s) for s in self.suppliers],
            "nodes": [asdict(n) for n in self.nodes],
            "in_transit": [asdict(sh) for sh in self.in_transit],
            "total_cost": round(self._total_cost, 2),
            "total_stockouts": self._total_stockouts,
            "total_delivered": self._total_delivered,
        }

    def _apply_disruptions(self):
        """Randomly toggle supplier availability based on reliability."""
        if self.difficulty == "easy":
            return
        for s in self.suppliers:
            roll = self._rng.random()
            if s.active and roll > s.reliability:
                s.active = False
            elif not s.active and roll < 0.4:
                s.active = True  # recover

        # Hard mode: demand spikes
        if self.difficulty == "hard" and self._rng.random() < 0.15:
            spike_node = self._rng.choice(self.nodes)
            spike_node.demand_per_step = min(
                int(spike_node.demand_per_step * 1.3),
                spike_node.capacity // 4
            )

    def _advance_shipments(self):
        """Tick down ETAs, deliver arrived shipments."""
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
            # Distribute to nodes that need it most
            needed = sorted(self.nodes, key=lambda n: n.inventory)
            qty = ship.quantity
            for node in needed:
                space = node.capacity - node.inventory
                add = min(space, qty)
                node.inventory += add
                self._total_delivered += add
                qty -= add
                if qty <= 0:
                    break

    def _consume_demand(self) -> float:
        """Consume demand at each node, track stockouts."""
        penalty = 0.0
        for node in self.nodes:
            demand = node.demand_per_step + node.backlog
            if node.inventory >= demand:
                node.inventory -= demand
                node.backlog = 0
            else:
                shortfall = demand - node.inventory
                node.backlog = shortfall
                node.inventory = 0
                penalty += shortfall * 10.0   # ₹10 penalty per unit stockout
                self._total_stockouts += shortfall
        return penalty

    # ── Public OpenEnv API ───────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset environment to initial state. Returns initial observation."""
        self._rng = random.Random(self.seed)
        self._setup_scenario()
        return self._encode_state()

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        """
        Take one step in the environment.

        Action format:
          {
            "type": "order" | "expedite" | "wait" | "reroute",
            "supplier_id": "S1",    # for order/expedite
            "quantity": 30,         # for order/expedite
            "from_node": "W1",      # for reroute
            "to_node": "W2",        # for reroute
            "transfer_qty": 10      # for reroute
          }

        Returns: (observation, reward, done, info)
        """
        self._step_count += 1
        reward = 0.0
        info = {"action_taken": action.get("type", "unknown"), "events": []}

        # ── Apply disruptions ──
        self._apply_disruptions()

        # ── Process action ──
        action_type = action.get("type", "wait")

        if action_type == "order":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity", 0))
            supplier = next((s for s in self.suppliers if s.id == sid), None)
            if supplier and supplier.active and qty > 0:
                cost = supplier.cost_per_unit * qty
                self._total_cost += cost
                reward -= cost * 0.1   # small cost signal
                self.in_transit.append(Shipment(sid, qty, supplier.lead_time))
                info["events"].append(f"Ordered {qty} units from {supplier.name} (cost ${cost:.2f})")
            elif supplier and not supplier.active:
                reward -= 5.0          # penalty for ordering from disrupted supplier
                info["events"].append(f"Supplier {sid} is disrupted!")

        elif action_type == "expedite":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity", 0))
            supplier = next((s for s in self.suppliers if s.id == sid), None)
            if supplier and supplier.active and qty > 0:
                cost = supplier.cost_per_unit * qty * 1.8  # 80% premium
                self._total_cost += cost
                reward -= cost * 0.1
                self.in_transit.append(Shipment(sid, qty, 1))  # arrive next step
                info["events"].append(f"Expedited {qty} units from {supplier.name} (cost ${cost:.2f})")

        elif action_type == "reroute":
            from_id = action.get("from_node")
            to_id   = action.get("to_node")
            qty     = int(action.get("transfer_qty", 0))
            from_n  = next((n for n in self.nodes if n.id == from_id), None)
            to_n    = next((n for n in self.nodes if n.id == to_id), None)
            if from_n and to_n and qty > 0:
                actual = min(qty, from_n.inventory)
                space  = to_n.capacity - to_n.inventory
                moved  = min(actual, space)
                from_n.inventory -= moved
                to_n.inventory   += moved
                reward -= moved * 0.5   # transfer cost
                info["events"].append(f"Rerouted {moved} units from {from_id} → {to_id}")

        # action_type == "wait" → no action cost

        # ── Advance shipments ──
        self._advance_shipments()

        # ── Consume demand ──
        stockout_penalty = self._consume_demand()
        reward -= stockout_penalty

        # ── Reward for good inventory health ──
        for node in self.nodes:
            fill_ratio = node.inventory / node.capacity
            if 0.2 <= fill_ratio <= 0.8:
                reward += 1.0   # healthy buffer bonus

        # ── Done? ──
        done = self._step_count >= self._max_steps

        obs = self._encode_state()
        return obs, round(reward, 4), done, info

    def state(self) -> dict:
        """Return current state without advancing the environment."""
        return self._encode_state()

    def action_space(self) -> dict:
        """Describe the action space."""
        return {
            "type": "discrete_parametric",
            "actions": [
                {
                    "type": "order",
                    "params": {
                        "supplier_id": [s.id for s in self.suppliers],
                        "quantity": {"min": 1, "max": 100}
                    }
                },
                {
                    "type": "expedite",
                    "params": {
                        "supplier_id": [s.id for s in self.suppliers],
                        "quantity": {"min": 1, "max": 50}
                    }
                },
                {
                    "type": "reroute",
                    "params": {
                        "from_node": [n.id for n in self.nodes],
                        "to_node":   [n.id for n in self.nodes],
                        "transfer_qty": {"min": 1, "max": 50}
                    }
                },
                {"type": "wait"}
            ]
        }

    def observation_space(self) -> dict:
        """Describe the observation space."""
        return {
            "type": "dict",
            "fields": {
                "step":              "int",
                "max_steps":         "int",
                "difficulty":        "str",
                "suppliers":         "list[Supplier]",
                "nodes":             "list[Node]",
                "in_transit":        "list[Shipment]",
                "total_cost":        "float",
                "total_stockouts":   "int",
                "total_delivered":   "int",
            }
        }
