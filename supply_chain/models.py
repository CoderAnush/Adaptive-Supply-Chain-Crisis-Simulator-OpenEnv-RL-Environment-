from pydantic import BaseModel, Field
from typing import Dict, List, Optional

class Shipment(BaseModel):
    id: str
    source: str
    destination: str
    quantity: int
    transport_mode: str
    eta: int
    cost: float
    delay: int = 0

class Crisis(BaseModel):
    id: str
    type: str # "Storm", "PortStrike", "DemandSpike"
    affected_node_or_edge: str
    severity: float
    duration: int
    probability: float

class Observation(BaseModel):
    inventories: Dict[str, int]
    shipments_in_transit: List[Shipment]
    active_crises: List[Crisis]
    known_delays: Dict[str, int]
    current_demand: Dict[str, int]
    step_count: int

class State(BaseModel):
    inventories: Dict[str, int]
    shipments_in_transit: List[Shipment]
    active_crises: List[Crisis]
    full_demand: Dict[str, int] # Hidden from agent
    step_count: int
    max_steps: int
    total_cost: float
    total_fulfilled: int
    total_unfulfilled: int
    historical_demand: Dict[str, List[int]] = Field(default_factory=dict)

class RouteAction(BaseModel):
    source: str
    destination: str
    quantity: int
    transport_mode: str # "Air" or "Sea"

class Action(BaseModel):
    routes: List[RouteAction]

class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float]
