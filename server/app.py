import os
import uvicorn
from fastapi import FastAPI, Request
from supply_chain.env import SupplyChainEnv
from supply_chain.models import Action

# --- OpenEnv FastAPI Server ---
# This server implements the required OpenEnv REST API for remote agent evaluation.
app = FastAPI(title="Supply Chain Crisis Simulator")
default_env = SupplyChainEnv()

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Adaptive Global Supply Chain Crisis Simulator is running.",
        "spec": "OpenEnv v1.0",
        "endpoints": ["/reset (POST)", "/step (POST)", "/state (GET)"]
    }

@app.post("/reset")
def api_reset():
    # Resets the simulation to the initial state
    obs = default_env.reset()
    return obs.dict()

@app.post("/step")
async def api_step(request: Request):
    # Process an action and return the resulting observation and reward
    try:
        body = await request.json()
        action_obj = Action(**body)
    except Exception:
        action_obj = Action(routes=[]) # fallback to empty action on parse error
        
    obs, reward, done, info = default_env.step(action_obj)
    return {
        "observation": obs.dict(),
        "reward": reward.dict(),
        "done": done,
        "info": info
    }

@app.get("/state")
def api_state():
    # Returns the full internal state (for debugging or state-aware agents)
    return default_env.state().dict()

def main():
    # Launch uvicorn server on the port required by Hugging Face Spaces
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
