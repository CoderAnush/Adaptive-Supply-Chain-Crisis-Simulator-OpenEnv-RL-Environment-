from typing import Dict, Any
from .models import State, Reward

def compute_reward(state: State, fulfilled: int, unfulfilled_added: int, transport_costs: float) -> Reward:
    FULFILLMENT_WEIGHT = 5.0
    TRANSPORT_COST_WEIGHT = -0.5
    HOLDING_COST_WEIGHT = -0.1
    UNFULFILLED_PENALTY_WEIGHT = -2.0

    holding_costs = 0.0
    for node, inv in state.inventories.items():
        if "Warehouse" in node:
            holding_costs += float(inv)

    fulfillment_reward = fulfilled * FULFILLMENT_WEIGHT
    transport_penalty = transport_costs * TRANSPORT_COST_WEIGHT
    holding_penalty = holding_costs * HOLDING_COST_WEIGHT
    unfulfilled_penalty = unfulfilled_added * UNFULFILLED_PENALTY_WEIGHT

    total_value = fulfillment_reward + transport_penalty + holding_penalty + unfulfilled_penalty
    normalized_value = total_value / 100.0

    breakdown = {
        "fulfillment_reward": fulfillment_reward,
        "transport_costs": transport_penalty,
        "holding_costs": holding_penalty,
        "unfulfilled_penalty": unfulfilled_penalty
    }
    return Reward(value=normalized_value, breakdown=breakdown)
