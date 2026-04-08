import os
import uvicorn
from fastapi import FastAPI, Request
from supply_chain.env import SupplyChainEnv
from supply_chain.graders import TASKS
from supply_chain.models import Action, RouteAction
from supply_chain.agent import LLMAgent

def run_agent(env, max_steps, agent=None, debug=False):
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        if agent:
            action = agent.get_action(obs)
        else:
            # Default heuristic agent implementation
            routes = []
            
            # 1. Fulfill highest demand first
            demand_na = obs.current_demand.get("Market_NA", 0)
            demand_eu = obs.current_demand.get("Market_EU", 0)
            
            inventory_us = obs.inventories.get("Warehouse_US", 0)
            inventory_eu = obs.inventories.get("Warehouse_EU", 0)
            
            # Send from warehouses to markets (Trucks)
            if demand_na > 0 and inventory_us > 0:
                qty = min(inventory_us, demand_na)
                routes.append(RouteAction(source="Warehouse_US", destination="Market_NA", quantity=qty, transport_mode="Truck"))
            if demand_eu > 0 and inventory_eu > 0:
                qty = min(inventory_eu, demand_eu)
                routes.append(RouteAction(source="Warehouse_EU", destination="Market_EU", quantity=qty, transport_mode="Truck"))
                
            # 2. Calculate pending stock to avoid over-ordering
            pending_us = sum(s.quantity for s in obs.shipments_in_transit if s.destination == "Warehouse_US")
            pending_eu = sum(s.quantity for s in obs.shipments_in_transit if s.destination == "Warehouse_EU")
            
            # Keep a buffer based on current demand
            target_us = demand_na * 2 + 10
            target_eu = demand_eu * 2 + 10
            
            deficit_us = max(0, target_us - (inventory_us + pending_us))
            deficit_eu = max(0, target_eu - (inventory_eu + pending_eu))
            
            fac_asia = obs.inventories.get("Factory_Asia", 0)
            fac_eu = obs.inventories.get("Factory_Europe", 0)
            
            # 3. Use Sea if time allows, Air if urgent
            if deficit_us > 0 and fac_asia > 0:
                mode = "Air" if inventory_us == 0 and demand_na > 0 else "Sea"
                qty = min(deficit_us, fac_asia)
                routes.append(RouteAction(source="Factory_Asia", destination="Warehouse_US", quantity=qty, transport_mode=mode))
                
            if deficit_eu > 0 and fac_eu > 0:
                mode = "Air" if inventory_eu == 0 and demand_eu > 0 else "Sea"
                qty = min(deficit_eu, fac_eu)
                routes.append(RouteAction(source="Factory_Europe", destination="Warehouse_EU", quantity=qty, transport_mode=mode))
            
            action = Action(routes=routes)

        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        
        if debug:
            print(f"\n{'='*60}")
            print(f"Step {obs.step_count}/{max_steps} | Score so far: {total_reward:.2f}")
            print(f"{'-'*60}")
            print(f"[ Demand ] Market_NA: {obs.current_demand.get('Market_NA', 0):>3} | Market_EU: {obs.current_demand.get('Market_EU', 0):>3}")
            print(f"[ Supply ] Asia: {obs.inventories.get('Factory_Asia', 0):>4}   | EU Factory: {obs.inventories.get('Factory_Europe', 0):>4}")
            print(f"[ Buffer ] Whse_US: {obs.inventories.get('Warehouse_US', 0):>3} | Whse_EU: {obs.inventories.get('Warehouse_EU', 0):>3}")
            if obs.active_crises:
                print(f"[ Crises ] {[c.type + ' on ' + c.affected_node_or_edge for c in obs.active_crises]}")
            
            action_strs = [f"{r.quantity}x via {r.transport_mode} to {r.destination}" for r in action.routes]
            if not action_strs: action_strs = ["Hold"]
            print(f"[ Action ] {', '.join(action_strs)}")
            print(f"{'='*60}")
            
    return total_reward

def evaluate_tasks():
    results = {}
    print("Evaluating Global Supply Chain Environment with OpenAI Baseline...")
    
    # Initialize LLM Agent as required by specifications
    agent = LLMAgent(model=os.getenv("OPENAI_MODEL", "gpt-4o"))
    
    for task_name, grader in TASKS.items():
        env = SupplyChainEnv(config=grader.config)
        # LLM based decision making
        run_agent(env, grader.config["max_steps"], agent=agent, debug=False)
        score = grader.grade(env)
        results[task_name] = round(score, 2)
        print(f"Task: {task_name.upper()} | Baseline Score: {score:.2f}")
    return results

# --- OpenEnv FastAPI Server ---
app = FastAPI(title="Supply Chain Crisis Simulator")
default_env = SupplyChainEnv()

@app.get("/")
def read_root():
    return {
        "Hello": "World!",
        "status": "online",
        "message": "Adaptive Global Supply Chain Crisis Simulator is running.",
        "endpoints": ["/reset (POST)", "/step (POST)", "/state (GET)", "/evaluate (GET)"]
    }

@app.get("/evaluate")
def api_evaluate():
    scores = evaluate_tasks()
    return {"baseline_scores": scores}

@app.post("/reset")
def api_reset():
    obs = default_env.reset()
    return obs.dict()

@app.post("/step")
async def api_step(request: Request):
    try:
        body = await request.json()
        action_obj = Action(**body)
    except Exception as e:
        action_obj = Action(routes=[]) # fallback/empty action
        
    obs, reward, done, info = default_env.step(action_obj)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def api_state():
    return default_env.state().dict()

if __name__ == "__main__":
    # 1. Run baseline evaluation
    evaluate_tasks()
    
    # 2. Start the HTTP server required by Hugging Face Spaces (Port 7860)
    print("\nStarting OpenEnv Server on port 7860...")
    uvicorn.run(app, host="0.0.0.0", port=7860)
