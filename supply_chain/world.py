import networkx as nx
from typing import List, Dict, Tuple, Optional
from .models import Shipment, RouteAction, State
import uuid

class SupplyChainWorld:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.build_graph()

    def build_graph(self):
        """Builds a realistic directed graph of a global supply chain."""
        self.graph.add_node("Factory_Asia", type="factory")
        self.graph.add_node("Factory_Europe", type="factory")
        self.graph.add_node("Warehouse_US", type="warehouse")
        self.graph.add_node("Warehouse_EU", type="warehouse")
        self.graph.add_node("Market_NA", type="market")
        self.graph.add_node("Market_EU", type="market")

        # Edges
        routes = [
            ("Factory_Asia", "Warehouse_US", "Sea", 2.0, 5),
            ("Factory_Asia", "Warehouse_US", "Air", 10.0, 2),
            ("Factory_Asia", "Warehouse_EU", "Sea", 1.5, 4),
            ("Factory_Asia", "Warehouse_EU", "Air", 8.0, 2),
            ("Factory_Europe", "Warehouse_EU", "Sea", 0.5, 1),
            ("Factory_Europe", "Warehouse_US", "Sea", 1.5, 4),
            ("Warehouse_US", "Market_NA", "Truck", 0.2, 1),
            ("Warehouse_EU", "Market_EU", "Truck", 0.2, 1)
        ]

        for src, dst, mode, cost, time in routes:
            self.graph.add_edge(src, dst, key=mode, cost=cost, time=time, base_time=time, delay=0)

    def get_valid_routes(self, source: str, destination: str) -> List[Dict]:
        routes = []
        if self.graph.has_edge(source, destination):
            for key, data in self.graph[source][destination].items():
                routes.append({"mode": key, "cost": data["cost"], "time": data["base_time"] + data["delay"]})
        return routes

    def dispatch_shipment(self, action: RouteAction, state: State) -> Tuple[bool, Optional[Shipment]]:
        if not self.graph.has_edge(action.source, action.destination) or action.transport_mode not in self.graph[action.source][action.destination]:
            return False, None
            
        edge_data = self.graph[action.source][action.destination][action.transport_mode]
        travel_time = edge_data["base_time"] + edge_data.get("delay", 0)
        cost = edge_data["cost"] * action.quantity

        shipment = Shipment(
            id=str(uuid.uuid4())[:8],
            source=action.source,
            destination=action.destination,
            quantity=action.quantity,
            transport_mode=action.transport_mode,
            eta=state.step_count + travel_time,
            cost=cost,
            delay=edge_data.get("delay", 0)
        )
        return True, shipment

    def deliver_shipments(self, state: State) -> List[Shipment]:
        delivered = []
        in_transit = []
        
        for shipment in state.shipments_in_transit:
            current_delay = 0
            if self.graph.has_edge(shipment.source, shipment.destination) and shipment.transport_mode in self.graph[shipment.source][shipment.destination]:
                current_delay = self.graph[shipment.source][shipment.destination][shipment.transport_mode].get("delay", 0)
            
            if current_delay > shipment.delay:
                shipment.eta += (current_delay - shipment.delay)
                shipment.delay = current_delay

            if state.step_count >= shipment.eta:
                delivered.append(shipment)
            else:
                in_transit.append(shipment)
                
        state.shipments_in_transit = in_transit
        return delivered
