from pydantic import BaseModel, Field
from typing import Optional, List

# ─── ACTION ────────────────────────────────────────────────
# This is what the AI agent SENDS to the environment
class GrainStoreAction(BaseModel):
    action_type: str = Field(
        ...,
        description="One of: ventilate, drain, alert, inspect, ignore"
    )
    target_zone: str = Field(
        ...,
        description="Zone to act on: zone_A, zone_B, zone_C"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Why the agent chose this action"
    )

# ─── OBSERVATION ───────────────────────────────────────────
# This is what the environment SENDS BACK to the agent
class GrainStoreObservation(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    gas_level: float = Field(..., description="MQ135 gas reading in ppm")
    fill_level: float = Field(..., description="Silo fill level percentage")
    vibration: bool = Field(..., description="Vibration detected true/false")
    task_id: str = Field(..., description="Current task: task1, task2, task3")
    task_description: str = Field(..., description="What the agent must do")
    step_count: int = Field(..., description="How many steps taken so far")
    message: str = Field(..., description="Feedback message to agent")

# ─── REWARD ────────────────────────────────────────────────
# Score given after each action
class GrainStoreReward(BaseModel):
    value: float = Field(..., description="Score from 0.0 to 1.0")
    reason: str = Field(..., description="Why this score was given")

# ─── STATE ─────────────────────────────────────────────────
# Internal state of the environment
class GrainStoreState(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    total_reward: float
    done: bool
    sensor_data: dict