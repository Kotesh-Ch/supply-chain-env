---
title: Supply Chain MV
emoji: 🌖
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

title: Supply Chain Disruption Manager
emoji: 🏭
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# [Open Live App]:
https://kotesh-ch-supply-chain-mv.hf.space/docs

# 🏭 Supply Chain Disruption Manager

**OpenEnv Round 1 Submission** — Meta × Scaler OpenEnv Hackathon

An RL environment where an AI agent manages a multi-node supply chain network, handling real-world complexities like supplier disruptions, inventory imbalances, demand spikes, and cost trade-offs.

---

## 📌 Problem Statement

Modern supply chains are brittle. A single supplier failure can cascade into stockouts, lost revenue, and delivery delays. This environment challenges an AI agent to:

- Monitor inventory levels across multiple warehouse nodes
- Order from suppliers with varying reliability, cost, and lead times
- Reroute inventory between nodes to cover shortfalls
- Expedite shipments during crises (at a premium)
- Maintain service levels through cascading disruptions

---

## 🌍 Action Space

| Action | Parameters | Description |
|---|---|---|
| `order` | `supplier_id`, `quantity` | Place a standard order (arrives after lead_time steps) |
| `expedite` | `supplier_id`, `quantity` | Rush order (arrives next step, 80% cost premium) |
| `reroute` | `from_node`, `to_node`, `transfer_qty` | Transfer inventory between warehouses |
| `wait` | — | Do nothing this step |

**Example action:**
```json
{
  "type": "order",
  "supplier_id": "S1",
  "quantity": 30
}
```

---

## 👁️ Observation Space

```json
{
  "step": 12,
  "max_steps": 50,
  "difficulty": "medium",
  "suppliers": [
    {"id": "S1", "name": "AlphaSupply", "active": true, "reliability": 0.85,
     "lead_time": 2, "cost_per_unit": 5.0},
    {"id": "S2", "name": "BetaLogistics", "active": false, "reliability": 0.75,
     "lead_time": 3, "cost_per_unit": 4.0}
  ],
  "nodes": [
    {"id": "W1", "inventory": 45, "capacity": 120, "demand_per_step": 10, "backlog": 0},
    {"id": "W2", "inventory": 8,  "capacity": 100, "demand_per_step": 8,  "backlog": 2}
  ],
  "in_transit": [
    {"supplier_id": "S1", "quantity": 30, "eta": 1}
  ],
  "total_cost": 450.0,
  "total_stockouts": 5,
  "total_delivered": 120
}
```

## 📈 Baseline Performance

| Difficulty | Score |
|----------|------|
| Easy     | 0.74 |
| Medium   | 0.70 |
| Hard     | 0.48 |

Average Score: **0.64**

---

## 🎯 Tasks & Graders

| Task | Difficulty | Nodes | Suppliers | Max Steps | Disruptions |
|---|---|---|---|---|---|
| Single-Node Supply Chain | Easy | 1 | 1 | 30 | None |
| Multi-Node Network | Medium | 3 | 2 | 50 | Random failures |
| Cascading Disruptions | Hard | 5 | 4 | 80 | Cascading + demand spikes |

**Scoring formula:**
```
score = 0.4 × service_level + 0.3 × cost_score + 0.3 × reward_score
```
All scores normalised to [0.0, 1.0]. Minimum passing score: 0.4 per task.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+
- Docker
- Git + GitHub / HuggingFace account

### Install

```bash
git clone https://github.com/YOUR_USERNAME/supply-chain-env
cd supply-chain-env

pip install -r requirements.txt
```

### Run locally

```bash
# Start the server
uv run server

# Or directly:
uvicorn server:app --host 0.0.0.0 --port 7860
```

### Test the API

```bash
# Health check
curl http://localhost:7860/health

# Reset to medium difficulty
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium", "seed": 42}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"type": "order", "supplier_id": "S1", "quantity": 30}}'

# Get current state
curl http://localhost:7860/state
```

### Run inference script

```bash
# Direct mode (no server needed)
python inference.py --mode direct

# HTTP mode (server must be running)
python inference.py --mode http
```

### Run graders

```bash
python graders.py
```

### Pre-submission validation

```bash
python validate.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t supply-chain-env .

# Run
docker run -p 7860:7860 supply-chain-env

# Test health
curl http://localhost:7860/health
```

---

## ☁️ Deploy to HuggingFace Spaces

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Push (via openenv CLI)
openenv push --repo-id YOUR_USERNAME/supply-chain-env
```

---

## 📁 Project Structure

```
supply-chain-env/
├── environment.py   # Core OpenEnv environment (step/reset/state)
├── server.py        # FastAPI REST server
├── graders.py       # 3 task graders (easy/medium/hard)
├── inference.py     # Baseline inference script (REQUIRED)
├── validate.py      # Pre-submission validation
├── openenv.yaml     # OpenEnv spec file
├── Dockerfile       # HuggingFace Spaces deployment
├── requirements.txt # Python dependencies
└── README.md        # This file
```

---

## 🔑 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | API endpoint for LLM calls | `http://localhost:7860` |
| `MODEL_NAME` | Model identifier | `baseline-greedy` |
| `HF_TOKEN` | HuggingFace API token | `""` |

---

## 📊 Reward Design

| Event | Reward Signal |
|---|---|
| Unit stockout | −10.0 per unit |
| Ordering cost | −0.1 × cost |
| Expediting premium | −0.1 × (cost × 1.8) |
| Rerouting transfer | −0.5 per unit moved |
| Healthy inventory (20–80% fill) | +1.0 per node per step |
| Order from disrupted supplier | −5.0 flat penalty |

---

## 📜 License

MIT License — see [LICENSE](LICENSE)
4e07e26 (Supply Chain OpenEnv)
