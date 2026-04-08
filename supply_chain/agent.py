import os
import json
from openai import OpenAI
from .models import Action, Observation

class LLMAgent:
    def __init__(self, model="gpt-4o", api_key=None, base_url=None):
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: No API key found. LLMAgent will use a mock/heuristic approach.")
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        self.model = model

    def get_action(self, obs: Observation) -> Action:
        if not self.client:
            # Fallback to a simple heuristic if no API key is provided
            return self._heuristic_fallback(obs)

        prompt = f"""
        You are a Supply Chain Manager. Based on the following observation, decide on the best RouteActions.
        Your goal is to fulfill market demand while minimizing costs (Air is 10x more expensive than Sea) and avoiding inventory stockouts.
        
        Observation:
        {obs.json(indent=2)}
        
        Available Nodes: Factory_Asia, Factory_Europe, Warehouse_US, Warehouse_EU, Market_NA, Market_EU.
        Transport Modes: "Air", "Sea", "Truck" (Truck is only for Warehouse to Market).
        
        Return your decision as a JSON object matching the Action model (a list of routes).
        Example: {{"routes": [{{"source": "Factory_Asia", "destination": "Warehouse_US", "quantity": 50, "transport_mode": "Sea"}}]}}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" }
            )
            action_json = json.loads(response.choices[0].message.content)
            return Action(**action_json)
        except Exception as e:
            print(f"LLM Error: {e}")
            return self._heuristic_fallback(obs)

    def _heuristic_fallback(self, obs: Observation) -> Action:
        # Simplified version of the heuristic in app.py
        from .models import RouteAction
        routes = []
        demand_na = obs.current_demand.get("Market_NA", 0)
        inventory_us = obs.inventories.get("Warehouse_US", 0)
        if demand_na > 0 and inventory_us > 0:
            routes.append(RouteAction(source="Warehouse_US", destination="Market_NA", quantity=min(inventory_us, demand_na), transport_mode="Truck"))
        
        demand_eu = obs.current_demand.get("Market_EU", 0)
        inventory_eu = obs.inventories.get("Warehouse_EU", 0)
        if demand_eu > 0 and inventory_eu > 0:
            routes.append(RouteAction(source="Warehouse_EU", destination="Market_EU", quantity=min(inventory_eu, demand_eu), transport_mode="Truck"))
            
        return Action(routes=routes)
