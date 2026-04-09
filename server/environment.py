"""
server/environment.py — Core Supply Chain Disruption Manager environment.
Implements reset(), step(), state() following OpenEnv spec.
"""

import random
import math
from typing import Optional, Tuple
from dataclasses import dataclass, asdict, field


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


class SupplyChainEnvironment:
    """
    Supply Chain Disruption Manager — OpenEnv-compliant environment.

    Three tasks (difficulties):
      easy   — 1 supplier, 1 node, stable
      medium — 2 suppliers, 3 nodes, random disruptions
      hard   — 4 suppliers, 5 nodes, cascading disruptions + demand spikes
    """

    DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(self):
        self.difficulty = "easy"
        self._seed: Optional[int] = None
        self._rng = random.Random(None)
        self._step_count = 0
        self._max_steps = 30
        self._total_cost = 0.0
        self._total_stockouts = 0
        self._total_delivered = 0
        self._episode_reward = 0.0
        self.suppliers: list[Supplier] = []
        self.nodes: list[Node] = []
        self.in_transit: list[Shipment] = []
        self._setup_scenario("easy")

    # ── Private ───────────────────────────────────────────────────────────────

    def _setup_scenario(self, difficulty: str):
        self.difficulty = difficulty
        self._max_steps = {"easy": 30, "medium": 50, "hard": 80}[difficulty]
        self._step_count = 0
        self._total_cost = 0.0
        self._total_stockouts = 0
        self._total_delivered = 0
        self._episode_reward = 0.0
        self.in_transit = []

        if difficulty == "easy":
            self.suppliers = [Supplier("S1", "AlphaSupply", True, 0.95, 2, 5.0)]
            self.nodes = [Node("W1", 50, 100, 8)]

        elif difficulty == "medium":
            self.suppliers = [
                Supplier("S1", "AlphaSupply",   True, 0.85, 2, 5.0),
                Supplier("S2", "BetaLogistics",  True, 0.75, 3, 4.0),
            ]
            self.nodes = [
                Node("W1", 60, 120, 10),
                Node("W2", 40, 100, 8),
                Node("W3", 30, 80,  6),
            ]

        else:  # hard
            self.suppliers = [
                Supplier("S1", "AlphaSupply",   True, 0.80, 2, 5.0),
                Supplier("S2", "BetaLogistics", True, 0.70, 3, 4.0),
                Supplier("S3", "GammaFreight",  True, 0.65, 1, 7.0),
                Supplier("S4", "DeltaExpress",  True, 0.90, 4, 8.0),
            ]
            self.nodes = [
                Node("W1", 80, 200, 8),
                Node("W2", 60, 160, 7),
                Node("W3", 40, 140, 6),
                Node("W4", 30, 120, 5),
                Node("W5", 20, 100, 4),
            ]

    def _apply_disruptions(self):
        if self.difficulty == "easy":
            return
        for s in self.suppliers:
            roll = self._rng.random()
            if s.active and roll > s.reliability:
                s.active = False
            elif not s.active and roll < 0.4:
                s.active = True
        if self.difficulty == "hard" and self._rng.random() < 0.15:
            node = self._rng.choice(self.nodes)
            node.demand_per_step = min(
                int(node.demand_per_step * 1.3),
                node.capacity // 4
            )

    def _advance_shipments(self):
        arrived, remaining = [], []
        for ship in self.in_transit:
            ship.eta -= 1
            (arrived if ship.eta <= 0 else remaining).append(ship)
        self.in_transit = remaining
        for ship in arrived:
            qty = ship.quantity
            for node in sorted(self.nodes, key=lambda n: n.inventory):
                add = min(node.capacity - node.inventory, qty)
                node.inventory += add
                self._total_delivered += add
                qty -= add
                if qty <= 0:
                    break

    def _consume_demand(self) -> float:
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
                penalty += shortfall * 10.0
                self._total_stockouts += shortfall
        return penalty

    def _encode(self) -> dict:
        return {
            "step":             self._step_count,
            "max_steps":        self._max_steps,
            "difficulty":       self.difficulty,
            "suppliers":        [asdict(s) for s in self.suppliers],
            "nodes":            [asdict(n) for n in self.nodes],
            "in_transit":       [asdict(sh) for sh in self.in_transit],
            "total_cost":       round(self._total_cost, 2),
            "total_stockouts":  self._total_stockouts,
            "total_delivered":  self._total_delivered,
        }

    def _score(self, total_reward: float) -> float:
        delivered = self._total_delivered
        stockouts = self._total_stockouts
        total     = delivered + stockouts
        svc = delivered / total if total > 0 else 0.0
        expected_cost = self._max_steps * len(self.nodes) * 60.0 * (1 + len(self.nodes) * 0.1)
        cost_score = math.exp(-self._total_cost / (expected_cost + 1e-9) * 0.5)
        floor   = -(self._max_steps * len(self.nodes) * 15.0)
        ceiling =   self._max_steps * len(self.nodes) * 1.5
        reward_score = (total_reward - floor) / (ceiling - floor + 1e-9)
        return round(
            0.4 * max(0.0, min(1.0, svc)) +
            0.3 * max(0.0, min(1.0, cost_score)) +
            0.3 * max(0.0, min(1.0, reward_score)),
            4
        )

    # ── Public OpenEnv API ────────────────────────────────────────────────────

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> dict:
        """Reset environment and return initial observation."""
        if difficulty not in self.DIFFICULTIES:
            difficulty = "easy"
        self._seed = seed
        self._rng = random.Random(seed)
        self._setup_scenario(difficulty)
        obs = self._encode()
        obs["reward"] = 0.0
        obs["done"] = False
        obs["success"] = True
        obs["message"] = f"Environment reset. Difficulty: {difficulty}"
        return obs

    def step(self, action: dict) -> Tuple[dict, float, bool, dict]:
        """Take one step. Returns (obs, reward, done, info)."""
        self._step_count += 1
        reward = 0.0
        info = {"action": action.get("action_type", "wait"), "events": []}

        self._apply_disruptions()

        atype = action.get("action_type", "wait")

        if atype == "order":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity") or 0)
            supplier = next((s for s in self.suppliers if s.id == sid), None)
            if supplier and supplier.active and qty > 0:
                cost = supplier.cost_per_unit * qty
                self._total_cost += cost
                reward -= cost * 0.1
                self.in_transit.append(Shipment(sid, qty, supplier.lead_time))
                info["events"].append(f"Ordered {qty} from {supplier.name}")
            elif supplier and not supplier.active:
                reward -= 5.0
                info["events"].append(f"Supplier {sid} disrupted!")

        elif atype == "expedite":
            sid = action.get("supplier_id")
            qty = int(action.get("quantity") or 0)
            supplier = next((s for s in self.suppliers if s.id == sid), None)
            if supplier and supplier.active and qty > 0:
                cost = supplier.cost_per_unit * qty * 1.8
                self._total_cost += cost
                reward -= cost * 0.1
                self.in_transit.append(Shipment(sid, qty, 1))
                info["events"].append(f"Expedited {qty} from {supplier.name}")

        elif atype == "reroute":
            from_id = action.get("from_node")
            to_id   = action.get("to_node")
            qty     = int(action.get("transfer_qty") or 0)
            from_n  = next((n for n in self.nodes if n.id == from_id), None)
            to_n    = next((n for n in self.nodes if n.id == to_id), None)
            if from_n and to_n and qty > 0:
                moved = min(qty, from_n.inventory, to_n.capacity - to_n.inventory)
                from_n.inventory -= moved
                to_n.inventory   += moved
                reward -= moved * 0.5
                info["events"].append(f"Rerouted {moved} units {from_id}→{to_id}")

        self._advance_shipments()
        stockout_penalty = self._consume_demand()
        reward -= stockout_penalty

        for node in self.nodes:
            fill = node.inventory / node.capacity
            if 0.2 <= fill <= 0.8:
                reward += 1.0

        self._episode_reward += reward
        done = self._step_count >= self._max_steps

        obs = self._encode()
        obs["reward"] = round(reward, 4)
        obs["done"] = done
        obs["success"] = True
        obs["message"] = ""

        if done:
            score = self._score(self._episode_reward)
            info["score"] = score
            obs["message"] = f"Episode complete. Score={score}"

        return obs, round(reward, 4), done, info

    def state(self) -> dict:
        """Return current state without advancing."""
        s = self._encode()
        s["episode_reward"] = round(self._episode_reward, 4)
        return s