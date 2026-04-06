from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os

# So Python can find models.py in the parent folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import GrainStoreAction, GrainStoreObservation, GrainStoreReward
from server.environment import GrainStoreEnvironment

# ── Create FastAPI app ────────────────────────────────────
app = FastAPI(
    title="GrainStore-Env",
    description="Smart grain silo monitoring OpenEnv environment",
    version="1.0.0"
)

# ── Allow all origins (needed for HF Spaces) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Create one environment instance ──────────────────────
env = GrainStoreEnvironment()


# ════════════════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Check if server is alive."""
    return {"status": "ok", "env": "GrainStore-Env"}
@app.get("/")
def root():
    return {
        "name": "GrainStore-Env",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks"
    }

@app.post("/reset")
def reset(task_id: str = "task1"):
    """
    Start a new episode.
    task_id: task1 (easy), task2 (medium), task3 (hard)
    """
    try:
        obs = env.reset(task_id=task_id)
        return {
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
            "info": {"task_id": task_id}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: GrainStoreAction):
    """
    Agent takes one action.
    Returns observation, reward, done, info.
    """
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return current environment state."""
    try:
        s = env.state()
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "id": "task1",
                "name": "Sensor Alert Detection",
                "difficulty": "easy",
                "description": "Identify the hazardous sensor and respond correctly.",
                "max_steps": 1,
                "max_reward": 1.0
            },
            {
                "id": "task2",
                "name": "Intervention Selection",
                "difficulty": "medium",
                "description": "Choose the best intervention for the detected hazard.",
                "max_steps": 1,
                "max_reward": 1.0
            },
            {
                "id": "task3",
                "name": "Multi-Hazard Response",
                "difficulty": "hard",
                "description": "Handle 3 simultaneous sensor hazards in correct order.",
                "max_steps": 3,
                "max_reward": 3.0
            }
        ]
    }
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()