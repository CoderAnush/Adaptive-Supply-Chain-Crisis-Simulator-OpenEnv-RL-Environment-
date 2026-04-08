import random
import uuid
from typing import List
from .models import State, Crisis
from .world import SupplyChainWorld

def generate_crises(state: State, world: SupplyChainWorld, seed: int = None) -> List[Crisis]:
    if seed is not None:
        random.seed(seed + state.step_count)
    
    new_crises = []
    
    # Storm
    if random.random() < 0.15:
        edges = list(world.graph.edges(keys=True))
        sea_edges = [e for e in edges if e[2] == "Sea"]
        if sea_edges:
            affected_edge = random.choice(sea_edges)
            edge_name = f"{affected_edge[0]}->{affected_edge[1]}({affected_edge[2]})"
            new_crises.append(Crisis(
                id=str(uuid.uuid4())[:8],
                type="Storm",
                affected_node_or_edge=edge_name,
                severity=random.uniform(1.0, 3.0),
                duration=random.randint(2, 4),
                probability=1.0
            ))

    # PortStrike
    if random.random() < 0.05:
        warehouses = [n for n, attr in world.graph.nodes(data=True) if attr.get("type") == "warehouse"]
        if warehouses:
            affected_node = random.choice(warehouses)
            new_crises.append(Crisis(
                id=str(uuid.uuid4())[:8],
                type="PortStrike",
                affected_node_or_edge=affected_node,
                severity=random.uniform(3.0, 6.0),
                duration=random.randint(3, 5),
                probability=1.0
            ))

    # DemandSpike
    if random.random() < 0.10:
        markets = [n for n, attr in world.graph.nodes(data=True) if attr.get("type") == "market"]
        if markets:
            affected_market = random.choice(markets)
            new_crises.append(Crisis(
                id=str(uuid.uuid4())[:8],
                type="DemandSpike",
                affected_node_or_edge=affected_market,
                severity=random.uniform(1.5, 2.5),
                duration=random.randint(1, 3),
                probability=1.0
            ))

    return new_crises

def apply_crises(state: State, world: SupplyChainWorld):
    for u, v, k, d in world.graph.edges(data=True, keys=True):
        d["delay"] = 0

    active_crises = []
    for crisis in state.active_crises:
        crisis.duration -= 1
        if crisis.duration <= 0:
            continue
            
        active_crises.append(crisis)
        if crisis.type == "Storm":
            try:
                parts = crisis.affected_node_or_edge.split("->")
                src = parts[0]
                dst_mode = parts[1].split("(")
                dst, mode = dst_mode[0], dst_mode[1].replace(")", "")
                if world.graph.has_edge(src, dst) and mode in world.graph[src][dst]:
                    world.graph[src][dst][mode]["delay"] += int(crisis.severity)
            except:
                pass
        elif crisis.type == "PortStrike":
            node = crisis.affected_node_or_edge
            for u, v, k, d in world.graph.edges(data=True, keys=True):
                if v == node:
                    d["delay"] += int(crisis.severity)
                    
    state.active_crises = active_crises
