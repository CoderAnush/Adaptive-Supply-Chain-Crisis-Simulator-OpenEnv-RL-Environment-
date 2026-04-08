import asyncio
import os
import json
from typing import List, Optional
from openai import OpenAI
from supply_chain.env import SupplyChainEnv
from supply_chain.graders import TASKS
from supply_chain.models import Action
from supply_chain.agent import LLMAgent

# Constants as required by the grading specification
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
BENCHMARK = "supply_chain_simulator"

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

async def main():
    # Initialize the OpenAI-powered agent
    agent = LLMAgent(model=MODEL_NAME)
    
    for task_id, grader in TASKS.items():
        # Initialize environmental state for the specific task
        env = SupplyChainEnv(config=grader.config)
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        obs = env.reset()
        rewards: List[float] = []
        steps_taken = 0
        done = False
        
        # Standard OpenEnv loop
        for step in range(1, grader.config["max_steps"] + 1):
            if done:
                break
            
            # Agent makes a decision based on the observation
            action_obj = agent.get_action(obs)
            
            # Environment processes the action
            obs, reward, done, info = env.step(action_obj)
            
            error = info.get("error")
            rewards.append(reward.value)
            steps_taken = step
            
            # Logic to create a human-readable action string for logs
            action_desc = ", ".join([f"{r.quantity}x {r.transport_mode}" for r in action_obj.routes]) or "Hold"
            
            log_step(step=step, action=action_desc, reward=reward.value, done=done, error=error)
            
            if done:
                break
        
        # Final scoring using the task grader
        score = grader.grade(env)
        success = score >= 0.1 # Success threshold defined in specifications
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())
