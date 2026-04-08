import random
from typing import Tuple, Dict, Any
from .models import State, Observation, Action, Reward, RouteAction, Crisis
from .world import SupplyChainWorld
from .crisis import generate_crises, apply_crises
from .reward import compute_reward

class SupplyChainEnv:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.world = SupplyChainWorld()
        self.seed_val = self.config.get("seed", 42)
        random.seed(self.seed_val)
        self.internal_state = None

    def reset(self) -> Observation:
        random.seed(self.seed_val)
        self.world = SupplyChainWorld()
        inventories = {
            "Factory_Asia": 1000,
            "Factory_Europe": 500,
            "Warehouse_US": 50,
            "Warehouse_EU": 50,
            "Market_NA": 0,
            "Market_EU": 0
        }
        full_demand = {
            "Market_NA": random.randint(10, 30),
            "Market_EU": random.randint(10, 30)
        }
        self.internal_state = State(
            inventories=inventories,
            shipments_in_transit=[],
            active_crises=[],
            full_demand=full_demand,
            step_count=0,
            max_steps=self.config.get("max_steps", 20),
            total_cost=0.0,
            total_fulfilled=0,
            total_unfulfilled=0,
            historical_demand={"Market_NA": [], "Market_EU": []}
        )
        return self._get_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        # Robustness patch: evaluate raw dict payloads automatically from API agents
        if isinstance(action, dict):
            action = Action(**action)

        transport_costs_this_step = 0.0
        for route in action.routes:
            if self.internal_state.inventories.get(route.source, 0) >= route.quantity and route.quantity > 0:
                success, shipment = self.world.dispatch_shipment(route, self.internal_state)
                if success:
                    self.internal_state.inventories[route.source] -= route.quantity
                    self.internal_state.shipments_in_transit.append(shipment)
                    transport_costs_this_step += shipment.cost
                    self.internal_state.total_cost += shipment.cost

        self.internal_state.inventories["Factory_Asia"] += 50
        self.internal_state.inventories["Factory_Europe"] += 30

        apply_crises(self.internal_state, self.world)

        delivered = self.world.deliver_shipments(self.internal_state)
        for shipment in delivered:
            self.internal_state.inventories[shipment.destination] += shipment.quantity

        fulfilled_this_step = 0
        unfulfilled_this_step = 0
        for market in ["Market_NA", "Market_EU"]:
            available = self.internal_state.inventories[market]
            demand = self.internal_state.full_demand[market]
            self.internal_state.historical_demand[market].append(demand)
            
            if available >= demand:
                fulfilled_this_step += demand
                self.internal_state.inventories[market] -= demand
                self.internal_state.full_demand[market] = 0
            else:
                fulfilled_this_step += available
                missing = demand - available
                unfulfilled_this_step += missing
                self.internal_state.inventories[market] = 0
                self.internal_state.full_demand[market] = missing
                
            base_new_demand = random.randint(10, 30)
            multiplier = 1.0
            for c in self.internal_state.active_crises:
                if c.type == "DemandSpike" and c.affected_node_or_edge == market:
                    multiplier = c.severity
            self.internal_state.full_demand[market] += int(base_new_demand * multiplier)

        self.internal_state.total_fulfilled += fulfilled_this_step
        self.internal_state.total_unfulfilled += unfulfilled_this_step

        if self.config.get("enable_crises", True):
            new_crises = generate_crises(self.internal_state, self.world, self.seed_val)
            self.internal_state.active_crises.extend(new_crises)

        reward = compute_reward(self.internal_state, fulfilled_this_step, unfulfilled_this_step, transport_costs_this_step)
        self.internal_state.step_count += 1
        done = self.internal_state.step_count >= self.internal_state.max_steps

        info = {
            "fulfilled": fulfilled_this_step,
            "unfulfilled_added": unfulfilled_this_step,
            "transport_costs": transport_costs_this_step
        }

        return self._get_observation(), reward, done, info

    def state(self) -> State:
        return self.internal_state

    def _get_observation(self) -> Observation:
        current_demand_obs = {k: v for k, v in self.internal_state.full_demand.items()}
        known_delays = {}
        for u, v, k, d in self.world.graph.edges(data=True, keys=True):
            if d.get("delay", 0) > 0:
                known_delays[f"{u}->{v}({k})"] = d["delay"]

        return Observation(
            inventories=self.internal_state.inventories.copy(),
            shipments_in_transit=self.internal_state.shipments_in_transit.copy(),
            active_crises=[c for c in self.internal_state.active_crises if random.random() > 0.1],
            known_delays=known_delays,
            current_demand=current_demand_obs,
            step_count=self.internal_state.step_count
        )
