import asyncio
import os
import textwrap
import json
import traceback
from typing import List, Optional
from openai import OpenAI

# Domain specific imports
from supply_chain.env import SupplyChainEnv
from supply_chain.graders import TASKS
from supply_chain.agent import LLMAgent
from supply_chain.models import Action

# --- Mandatory Environment Configuration ---
# AS PER PLATFORM INSTRUCTIONS: 
# 1. Use square bracket access for os.environ["API_BASE_URL"] and os.environ["API_KEY"]
# 2. Initialize exactly as shown in the example
# 3. Use gpt-4o-mini as the default if MODEL_NAME is missing
API_BASE_URL = os.environ["API_BASE_URL"]
# The platform may provide the set variable as API_KEY, HF_TOKEN, or OPENENV_API_KEY
API_KEY = os.environ.get("OPENENV_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

print(f"[DEBUG] Initializing with LLM Proxy: {API_BASE_URL}")
print(f"[DEBUG] Model in use: {MODEL_NAME}")

if not API_KEY:
    print("[WARNING] No API_KEY or HF_TOKEN found in environment variables.", flush=True)

# Task selection: Priorities the environment variable from OpenEnv/Grader
TASK_ID = os.getenv("OPENENV_TASK_ID") or os.getenv("MY_ENV_V4_TASK") or os.getenv("TASK_NAME") or "steady_state"
BENCHMARK = "supply_chain_simulator"
SUCCESS_SCORE_THRESHOLD = 0.1

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

async def main() -> None:
    # Initialize the Agent
    agent = LLMAgent(model=MODEL_NAME, api_key=API_KEY, base_url=API_BASE_URL)
    
    # --- Mandatory Proxy Test Call ---
    # Ensures at least one call is detected by the platform's proxy
    print("[DEBUG] Performing proxy test call...", flush=True)
    try:
        test_client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        test_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": "Test call to verify proxy connectivity."}],
            max_tokens=5
        )
        print("[DEBUG] Proxy test call successful.", flush=True)
    except Exception as e:
        print(f"[ERROR] Proxy test call failed: {e}", flush=True)
        # We continue anyway, but this helps debug in the logs
    
    # Select the specific task
    grader = TASKS.get(TASK_ID)
    if not grader:
        # Default to steady_state if not found
        TASK_NAME = "steady_state"
        grader = TASKS[TASK_NAME]
    else:
        TASK_NAME = TASK_ID

    # Initialize the environment locally (as it is included in the package)
    env = SupplyChainEnv(config=grader.config)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        max_steps = grader.config.get("max_steps", 20)

        for step in range(1, max_steps + 1):
            try:
                # Agent generates action based on current observation
                action_obj = agent.get_action(obs)
                
                # Execute action in the environment
                obs, reward_obj, done, info = env.step(action_obj)
                
                reward = reward_obj.value
                error = info.get("error")
                
                rewards.append(reward)
                steps_taken = step
                
                # Format action description for logging
                action_desc = ", ".join([f"{r.quantity}x {r.transport_mode}" for r in action_obj.routes]) or "Hold"
                
                log_step(step=step, action=action_desc, reward=reward, done=done, error=error)
                
                if done:
                    break
            except Exception as e:
                print(f"[DEBUG] Step {step} error: {e}", flush=True)
                log_step(step=step, action="error", reward=0.0, done=True, error=str(e))
                break

        # Calculate final score and success state
        score = grader.grade(env)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Fatal execution error: {e}", flush=True)
        traceback.print_exc()
    finally:
        # Ensure log_end is always called as per specification
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:
        traceback.print_exc()
        os._exit(1)
