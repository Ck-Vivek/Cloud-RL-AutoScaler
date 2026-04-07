import os
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from models import CloudObservation, CloudAction, CloudReward
from env import CloudScalingEnv

app = FastAPI(title="Cloud-RL Auto-scaler OpenEnv API")

# Global environment instance
env_instance = None

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Cloud-RL OpenEnv FastAPI is running!"}

@app.post("/reset")
def reset():
    global env_instance
    # Re-initialize the environment
    env_instance = CloudScalingEnv(task_level="Medium")
    obs = env_instance.reset()
    # The automated grader expects the reset observation
    return obs.dict()

@app.post("/step")
def step(action: CloudAction):
    global env_instance
    if env_instance is None:
        env_instance = CloudScalingEnv(task_level="Medium")
        env_instance.reset()
    
    try:
        obs_new, rew, done = env_instance.step(action)
        return {
            "observation": obs_new.dict(),
            "reward": rew.dict(),
            "done": done,
            "info": {}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
