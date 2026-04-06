"""
GrainStore-Env Baseline Inference Script
Uses Groq (OpenAI-compatible) to run an LLM agent against all 3 tasks.
"""

import asyncio
import os
import json
from typing import List
from openai import OpenAI
from client import GrainStoreEnv
from models import GrainStoreAction

# ── Configuration ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY      = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS         = 3
MAX_TOTAL_REWARD  = 3.0
SUCCESS_THRESHOLD = 0.5

TASKS = ["task1", "task2", "task3"]

# ════════════════════════════════════════════════════════
#  LOGGING — must follow exact format for judges
# ════════════════════════════════════════════════════════
def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type": "START",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    print(json.dumps({
        "type": "STEP",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error
    }), flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "type": "END",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards
    }), flush=True)


# ════════════════════════════════════════════════════════
#  LLM AGENT
# ════════════════════════════════════════════════════════
def get_model_action(client: OpenAI, observation: dict, history: List[str]) -> GrainStoreAction:
    """Ask the LLM what action to take given sensor readings."""

    system_prompt = """You are an AI agent monitoring a smart grain silo.
You receive sensor readings and must choose the correct action to protect the grain.

SAFE RANGES:
- Temperature: 10-30°C  → if HIGH use: ventilate
- Humidity: 30-60%      → if HIGH use: drain
- Gas Level: 0-400 ppm  → if HIGH use: alert
- Fill Level: 10-90%    → if HIGH use: inspect
- Vibration: false      → if TRUE use: inspect
- All normal            → use: ignore

AVAILABLE ACTIONS: ventilate, drain, alert, inspect, ignore
TARGET ZONES: zone_A, zone_B, zone_C

You MUST respond with ONLY a valid JSON object like this:
{
  "action_type": "ventilate",
  "target_zone": "zone_A",
  "reasoning": "Temperature is 48.7C which is above safe range of 30C"
}

No explanation. No markdown. Just the JSON object."""

    # Build context from current observation
    user_message = f"""Current sensor readings:
- Temperature: {observation['temperature']}°C (safe: 10-30°C)
- Humidity: {observation['humidity']}% (safe: 30-60%)
- Gas Level: {observation['gas_level']} ppm (safe: 0-400 ppm)
- Fill Level: {observation['fill_level']}% (safe: 10-90%)
- Vibration: {observation['vibration']}

Task: {observation['task_description']}
Message: {observation['message']}

History: {' | '.join(history[-3:]) if history else 'None'}

What action do you take? Respond with JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=200,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message}
            ]
        )

        raw = response.choices[0].message.content.strip()

        # Clean markdown if LLM adds it
        raw = raw.replace("```json", "").replace("```", "").strip()
        depth = 0
        start = raw.find("{")
        end = start
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        raw = raw[start:end] 
        data = json.loads(raw)

        return GrainStoreAction(
            action_type=data.get("action_type", "inspect"),
            target_zone=data.get("target_zone") or "zone_A",
            reasoning=data.get("reasoning") or ""
        )

    except Exception as e:
        # Fallback safe action if LLM fails
        print(f"[DEBUG] LLM parse error: {e}", flush=True)
        return GrainStoreAction(
            action_type="inspect",
            target_zone="zone_A",
            reasoning="fallback"
        )


# ════════════════════════════════════════════════════════
#  RUN ONE TASK
# ════════════════════════════════════════════════════════
async def run_task(client: OpenAI, task_id: str) -> float:
    """Run one full episode for a given task. Returns score 0.0-1.0."""

    log_start(task=task_id, env="GrainStore-Env", model=MODEL_NAME)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    async with GrainStoreEnv(base_url=ENV_BASE_URL) as env:
        try:
            # Reset environment
            result = await env.reset(task_id=task_id)
            obs = result["observation"]
            done = result["done"]

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                # Get action from LLM
                action = get_model_action(client, obs, history)

                # Take step
                result = await env.step(action)
                obs    = result["observation"]
                reward = result["reward"]["value"] if result["reward"] else 0.0
                done   = result["done"]
                error  = None

                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=action.action_type,
                    reward=reward,
                    done=done,
                    error=error
                )

                history.append(
                    f"Step {step}: {action.action_type} → reward {reward:.2f}"
                )

                if done:
                    break

            # Calculate final score
            max_reward = 3.0 if task_id == "task3" else 1.0
            score   = sum(rewards) / max_reward
            score   = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_THRESHOLD

        except Exception as e:
            print(f"[DEBUG] Task error: {e}", flush=True)
            error = str(e)

        finally:
            log_end(
                success=success,
                steps=steps_taken,
                score=score,
                rewards=rewards
            )

    return score


# ════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════
async def main():
    print("[INFO] Starting GrainStore-Env Baseline Inference", flush=True)
    print(f"[INFO] Model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Environment: {ENV_BASE_URL}", flush=True)

    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    all_scores = []

    for task_id in TASKS:
        print(f"\n[INFO] Running {task_id}...", flush=True)
        score = await run_task(client, task_id)
        all_scores.append(score)
        print(f"[INFO] {task_id} score: {score:.3f}", flush=True)

    final_score = sum(all_scores) / len(all_scores)
    print(f"\n[RESULT] Average score across all tasks: {final_score:.3f}", flush=True)
    print(f"[RESULT] Per task: task1={all_scores[0]:.3f}, task2={all_scores[1]:.3f}, task3={all_scores[2]:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())