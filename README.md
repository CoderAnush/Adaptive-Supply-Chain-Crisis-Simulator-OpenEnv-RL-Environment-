---
title: Supply Chain Crisis Simulator
emoji: 📦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# Adaptive Global Supply Chain Crisis Simulator

> **Train AI agents to manage global supply chains under uncertainty.**

**The Problem:** Global supply chains are incredibly fragile. When crises strike—like port blockages, natural disasters, or sudden demand spikes—human operators face cognitive overload, leading to panicky rerouting and massive financial waste. Real companies lose billions annually due to reactive logistics management.

**What we built:** A production-grade, real-world `OpenEnv` reinforcement learning environment simulating a global supply chain network. It evaluates an AI agent's ability to balance multi-objective trade-offs (e.g., cheap, slow sea freight vs. expensive, fast air freight) to meet strict market demands.

**Why it is a hard benchmark:** This is not a simple pathfinding puzzle. We introduced **partial observability** (agents don't see the true latent demand or hidden transport delays) and **stochastic crises** (probabilistic storms and port strikes). Agents require deep temporal planning and Operations Research (OR) reasoning under strict uncertainty to succeed.

## �️ System Architecture

```text
[Factories]                 [Transit]                 [Markets]
 (Supply)                 (Warehouses)                (Demand)
    │                          │                          │
    ├─Sea (Cheap/Slow) ─▶ 📦 Warehouse_US ─Truck─▶ 🛒 Market_NA
 🏭 Factory_Asia               │                          │
    │                          │                          │
    ├─Air (Expensive/Fast) ─▶ 📦 Warehouse_EU ─Truck─▶ 🛒 Market_EU
 🏭 Factory_Europe
```
*Agents must dynamically route shipments while optimizing for cost, time, and hidden disruptions.*

## 🚀 Why This Environment?
Global supply chains are the backbone of the modern economy. Logistics managers face massive cognitive overload balancing costs, delays, inventory, and unpredictable crises (e.g., storms, port strikes, demand spikes) with incomplete information. 

**Why this is hard for AI:**
This environment requires true systemic decision-making out of reach for basic heuristics. An AI agent must demonstrate:
- **Temporal planning across multiple steps** (sending inventory days before it's actually requested).
- **Reasoning under uncertainty** (anticipating future bottlenecks by reacting to delayed crisis indicators).
- **Trade-off optimization** (sacrificing margin on expensive Air freight to avoid compounding late delivery penalties).

## 💡 Example Episode Walkthrough

**Step 6 / 20**
- *Event:* `Storm` crisis detected near `Warehouse_US`. Sea transit ETA rises from 3 to 5 actions.
- *Information:* Agent checks `inventories` and realizes `Market_NA` will deplete in 2 actions.
- *Action:* Agent issues a `RouteAction` from `Factory_Asia` using `Air` freight (ETA: 1) instead of `Sea`.
- *Result:* Transport cost spikes, but the agent successfully avoids the compounding `unfulfilled_penalty` SLA breach, maximizing the multi-objective reward.

## 🧠 OpenEnv Interface

### Observation Space
The agent sees a **partial view** of the world state. Hidden information includes true future demand and some transport delays (revealed only when shipments fail to arrive on time).

```python
class Observation(BaseModel):
    inventories: Dict[str, int]                    # Goods stored at each node
    shipments_in_transit: List[Shipment]           # Live packages en-route (ETA visible)
    active_crises: List[Crisis]                    # Detected crises affecting routing
    known_delays: Dict[str, int]                   # Observed transport delays
    current_demand: Dict[str, int]                 # Market demand this step (partial signal)
    step_count: int                                # Current timestep
```

**Example:**
```json
{
  "inventories": {"Factory_Asia": 850, "Warehouse_US": 12},
  "shipments_in_transit": [
    {"id": "ship_0", "source": "Factory_Asia", "destination": "Warehouse_US", 
     "quantity": 100, "transport_mode": "Sea", "eta": 2, "cost": 500.0}
  ],
  "active_crises": [
    {"id": "storm_0", "type": "Storm", "affected_node_or_edge": "Asia->US_Sea", "severity": 0.8, "duration": 3}
  ],
  "known_delays": {"Asia->US_Sea": 2},
  "current_demand": {"Market_NA": 55, "Market_EU": 37},
  "step_count": 1
}
```

### Action Space
The agent routes shipments by specifying source, destination, quantity, and transport mode.

```python
class Action(BaseModel):
    routes: List[RouteAction]

class RouteAction(BaseModel):
    source: str                    # Factory or Warehouse name
    destination: str               # Warehouse or Market name
    quantity: int                  # Units to ship
    transport_mode: str            # "Air" (fast/expensive) or "Sea" (slow/cheap)
```

**Example:**
```json
{
  "routes": [
    {"source": "Factory_Asia", "destination": "Warehouse_US", "quantity": 50, "transport_mode": "Air"},
    {"source": "Factory_Europe", "destination": "Market_EU", "quantity": 30, "transport_mode": "Sea"},
    {"source": "Warehouse_US", "destination": "Market_NA", "quantity": 25, "transport_mode": "Truck"}
  ]
}
```

### Reward Signal
Multi-objective continuous reward balancing fulfillment, cost, inventory holding, and demand penalties.

```python
class Reward(BaseModel):
    value: float                           # Total reward this step
    breakdown: Dict[str, float]            # Granular reward components
```

**Reward Composition:**
- `+fulfillment_reward`: Points for satisfying market demands on time.
- `-transport_cost_penalty`: Cost of Air freight (10x more expensive than Sea).
- `-holding_cost_penalty`: Penalty for excessive inventory (inefficient working capital).
- `-delay_penalty`: Penalty when transport ETAs exceed demand windows.
- `-unfulfilled_penalty`: Heavy penalty for any unmet demand (causes SLA breach).

**Example breakdown:**
```json
{
  "value": 2.34,
  "breakdown": {
    "fulfillment_reward": 5.0,
    "transport_cost_penalty": -1.2,
    "holding_cost_penalty": -0.8,
    "delay_penalty": -0.6,
    "unfulfilled_penalty": 0.0
  }
}
```

## 🧪 Tasks
1. **Steady State (`steady_state`)**: Predictable demand, no crises. Tests basic environment comprehension.
2. **Suez Blockage (`suez_blockage`)**: Fixed crises interrupt standard cheap routing. Tests adaptive re-routing to expensive but fast modes.
3. **Black Swan (`black_swan`)**: Stochastic, overlapping crises (strikes, weather, demand spikes). Tests deep planning and partial observability resolution.

---

## 📊 Baseline Performance

We provide a programmatic `Heuristic Agent` (see `inference.py`) that prioritizes high-demand markets, relies primarily on `Sea` freight for efficiency, and falls back to `Air` freight for critical deficits. 

| Model/Agent        | Easy (Steady State) | Medium (Suez Blockage) | Hard (Black Swan) |
|--------------------|---------------------|------------------------|-------------------|
| **Heuristic Agent**| `0.76`              | `0.64`                 | `0.43`            |

### Why does performance drop on Hard?
The drop in performance from Easy to Hard demonstrates the increasing difficulty of planning under **uncertainty** and **partial observability**. In the Hard task, demand spikes probabilistically and hidden delays are only manifested mid-flight. A rigid heuristic fails here, severely penalizing the agent for unfulfilled orders and excessive holding costs. A frontier LLM using chain-of-thought (CoT) reasoning is required to interpret early-warning signals and pre-emptively buffer inventory.

---

## � Pre-flight Checklist (Hugging Face / OpenEnv)
- [x] Environment deployed to HuggingFace Spaces.
- [x] Valid `openenv.yaml` schema passing `openenv validate`.
- [x] Works seamlessly on HF Space build `docker build && docker run`.
- [x] Standard `step()`, `reset()`, and `state()` loop responds instantly.

---

## �� Quickstart

### Option 1: Docker (Recommended for Hugging Face)
```bash
docker build -t supply-chain-env .
docker run -it -p 7860:7860 supply-chain-env
```
The environment will start the FastAPI server on `http://localhost:7860`.

### Option 2: Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run the baseline evaluation
# Run the baseline evaluation (Compliant with [START]/[STEP]/[END] format)
python inference.py

# Or start the server directly for API testing
python app.py
```

---

## 📋 Setup Instructions

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- Git

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/CoderAnush/Adaptive-Supply-Chain-Crisis-Simulator-OpenEnv-RL-Environment-
cd supply_chain_env
```

**2. Create a virtual environment (Python only):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### Running the Environment

**As a local baseline evaluation:**
```bash
python app.py
```
This will:
- Initialize the supply chain world
- Run the heuristic agent through all 3 tasks (Easy, Medium, Hard)
- Print beautiful terminal dashboard logs showing each step
- Report final baseline scores

Expected output:
```
Task: STEADY_STATE | Baseline Score: 0.76
Task: SUEZ_BLOCKAGE | Baseline Score: 0.64
Task: BLACK_SWAN | Baseline Score: 0.43
```

**As an API server (for external agents):**
```bash
# Start the server
python -c "from app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=7860)"
```

Then interact with the environment via REST API using curl or Python:

```bash
# Reset the environment
curl -X POST http://localhost:7860/reset

# Take a step with an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"routes": [{"source": "Factory_Asia", "destination": "Warehouse_US", "quantity": 50, "transport_mode": "Sea"}]}'

# Get the current state
curl -X GET http://localhost:7860/state
```

### Validation

To ensure the environment passes OpenEnv specification compliance:
```bash
pip install openenv-validator
openenv validate
```