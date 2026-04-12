import os
import uvicorn
from fastapi import FastAPI, Request
from supply_chain.env import SupplyChainEnv
from supply_chain.models import Action
from supply_chain.graders import TASKS

# --- OpenEnv FastAPI Server ---
# This server implements the required OpenEnv REST API v1 for remote agent evaluation.
app = FastAPI(title="Supply Chain Crisis Simulator")
default_env = SupplyChainEnv()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Adaptive Global Supply Chain Crisis Simulator is running.",
        "spec": "OpenEnv v1.0",
        "endpoints": ["/v1/tasks", "/reset", "/step", "/state"]
    }

@app.get("/v1/tasks")
def get_tasks():
    # Returns the list of tasks for the OpenEnv validator with mandatory 'tags'
    return [
        {
            "id": "task_easy",
            "name": "Steady State Management",
            "difficulty": "easy",
            "tags": ["eval"],
            "description": "Easy task with predictable demand and no crises."
        },
        {
            "id": "task_medium",
            "name": "Route Blockage Response",
            "difficulty": "medium",
            "tags": ["eval"],
            "description": "Medium task introducing fixed route blockages."
        },
        {
            "id": "task_hard",
            "name": "Black Swan Crisis",
            "difficulty": "hard",
            "tags": ["eval"],
            "description": "Hard task with extreme, stochastic multi-modal crises."
        }
    ]

@app.post("/reset")
def api_reset(request: Request = None):
    obs = default_env.reset()
    return obs.dict()

@app.post("/step")
async def api_step(request: Request):
    try:
        body = await request.json()
        if "action" in body:
            action_obj = Action(**body["action"])
        else:
            action_obj = Action(**body)
    except Exception:
        action_obj = Action(routes=[])
        
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

def main():
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
