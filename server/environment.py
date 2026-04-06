import uuid
import random
from models import (
    GrainStoreAction,
    GrainStoreObservation,
    GrainStoreReward,
    GrainStoreState,
)

class GrainStoreEnvironment:
    """
    Grain Silo Monitoring Environment.
    Simulates a smart hermetic grain silo with IoT sensors.
    Agent must detect and respond to hazardous sensor readings.
    """

    # ── Safe ranges for each sensor ──────────────────────────
    SAFE_RANGES = {
        "temperature": (10.0, 30.0),   # Celsius
        "humidity":    (30.0, 60.0),   # Percentage
        "gas_level":   (0.0,  400.0),  # PPM
        "fill_level":  (10.0, 90.0),   # Percentage
    }

    # ── Correct actions for each hazard ──────────────────────
    CORRECT_ACTIONS = {
        "high_temperature": "ventilate",
        "high_humidity":    "drain",
        "high_gas":         "alert",
        "overfill":         "inspect",
        "vibration":        "inspect",
        "normal":           "ignore",
    }

    def __init__(self):
        self._state = None
        self._task_id = "task1"
        self._scenario = None
        self.reset()

    # ────────────────────────────────────────────────────────
    def reset(self, task_id: str = "task1") -> GrainStoreObservation:
        """Start a fresh episode."""
        self._task_id = task_id
        self._state = GrainStoreState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            total_reward=0.0,
            done=False,
            sensor_data={},
        )
        self._scenario = self._generate_scenario(task_id)
        return self._build_observation("Episode started. Analyze sensors and act.")

    # ────────────────────────────────────────────────────────
    def step(self, action: GrainStoreAction):
        """Agent takes an action. Returns observation, reward, done, info."""

        if self._state.done:
            obs = self._build_observation("Episode already finished. Call reset.")
            return obs, GrainStoreReward(value=0.0, reason="Episode done"), True, {}

        self._state.step_count += 1

        # Grade the action
        reward = self._grade_action(action)
        self._state.total_reward += reward.value

        # Episode ends after:
        # task1 → 1 step, task2 → 1 step, task3 → 3 steps
        max_steps = 3 if self._task_id == "task3" else 1
        done = self._state.step_count >= max_steps
        self._state.done = done

        message = self._feedback_message(action, reward, done)
        obs = self._build_observation(message)

        return obs, reward, done, {"episode_id": self._state.episode_id}

    # ────────────────────────────────────────────────────────
    def state(self) -> GrainStoreState:
        """Return current internal state."""
        return self._state

    # ════════════════════════════════════════════════════════
    #  SCENARIO GENERATORS
    # ════════════════════════════════════════════════════════
    def _generate_scenario(self, task_id: str) -> dict:
        """
        task1 — ONE sensor in danger. Agent must identify it.
        task2 — ONE sensor in danger. Agent must pick correct action.
        task3 — THREE sensors in danger. Agent must handle all.
        """
        if task_id == "task1":
            hazard = random.choice([
                "high_temperature", "high_humidity",
                "high_gas", "overfill"
            ])
            sensors = self._make_sensors_with_hazard(hazard)
            return {"hazards": [hazard], "sensors": sensors}

        elif task_id == "task2":
            hazard = random.choice([
                "high_temperature", "high_humidity",
                "high_gas", "overfill", "vibration"
            ])
            sensors = self._make_sensors_with_hazard(hazard)
            return {"hazards": [hazard], "sensors": sensors}

        elif task_id == "task3":
            hazards = random.sample([
                "high_temperature", "high_humidity",
                "high_gas", "overfill", "vibration"
            ], 3)
            sensors = self._make_sensors_with_multiple_hazards(hazards)
            return {"hazards": hazards, "sensors": sensors}

        return {}

    # ────────────────────────────────────────────────────────
    def _make_sensors_with_hazard(self, hazard: str) -> dict:
        """Normal sensor readings with one hazard injected."""
        sensors = {
            "temperature": round(random.uniform(15, 28), 1),
            "humidity":    round(random.uniform(35, 55), 1),
            "gas_level":   round(random.uniform(100, 350), 1),
            "fill_level":  round(random.uniform(20, 80), 1),
            "vibration":   False,
        }
        if hazard == "high_temperature":
            sensors["temperature"] = round(random.uniform(38, 55), 1)
        elif hazard == "high_humidity":
            sensors["humidity"] = round(random.uniform(75, 95), 1)
        elif hazard == "high_gas":
            sensors["gas_level"] = round(random.uniform(600, 1000), 1)
        elif hazard == "overfill":
            sensors["fill_level"] = round(random.uniform(92, 99), 1)
        elif hazard == "vibration":
            sensors["vibration"] = True
        return sensors

    # ────────────────────────────────────────────────────────
    def _make_sensors_with_multiple_hazards(self, hazards: list) -> dict:
        """Normal readings with multiple hazards injected."""
        sensors = {
            "temperature": round(random.uniform(15, 28), 1),
            "humidity":    round(random.uniform(35, 55), 1),
            "gas_level":   round(random.uniform(100, 350), 1),
            "fill_level":  round(random.uniform(20, 80), 1),
            "vibration":   False,
        }
        for hazard in hazards:
            if hazard == "high_temperature":
                sensors["temperature"] = round(random.uniform(38, 55), 1)
            elif hazard == "high_humidity":
                sensors["humidity"] = round(random.uniform(75, 95), 1)
            elif hazard == "high_gas":
                sensors["gas_level"] = round(random.uniform(600, 1000), 1)
            elif hazard == "overfill":
                sensors["fill_level"] = round(random.uniform(92, 99), 1)
            elif hazard == "vibration":
                sensors["vibration"] = True
        return sensors

    # ════════════════════════════════════════════════════════
    #  GRADERS
    # ════════════════════════════════════════════════════════
    def _grade_action(self, action: GrainStoreAction) -> GrainStoreReward:
        """Score the agent's action 0.0 to 1.0."""

        hazards = self._scenario["hazards"]

        if self._task_id == "task1":
            return self._grade_task1(action, hazards)
        elif self._task_id == "task2":
            return self._grade_task2(action, hazards)
        elif self._task_id == "task3":
            return self._grade_task3(action, hazards)

        return GrainStoreReward(value=0.0, reason="Unknown task")

    # ── Task 1 Grader: Did agent identify the right hazard? ──
    def _grade_task1(self, action, hazards) -> GrainStoreReward:
        hazard = hazards[0]
        correct_action = self.CORRECT_ACTIONS[hazard]

        if action.action_type == correct_action:
            return GrainStoreReward(
                value=1.0,
                reason=f"Correct! {hazard} requires {correct_action}."
            )
        elif action.action_type == "inspect":
            # Partial credit for being cautious
            return GrainStoreReward(
                value=0.3,
                reason=f"Partial: inspect is cautious but {correct_action} was needed."
            )
        else:
            return GrainStoreReward(
                value=0.0,
                reason=f"Wrong. {hazard} needed {correct_action}, got {action.action_type}."
            )

    # ── Task 2 Grader: Did agent pick the right intervention? ─
    def _grade_task2(self, action, hazards) -> GrainStoreReward:
        hazard = hazards[0]
        correct_action = self.CORRECT_ACTIONS[hazard]

        if action.action_type == correct_action:
            return GrainStoreReward(
                value=1.0,
                reason=f"Perfect intervention for {hazard}."
            )
        elif action.action_type == "alert":
            # Alerting is always somewhat reasonable
            return GrainStoreReward(
                value=0.4,
                reason=f"Alert is safe but {correct_action} was the best action."
            )
        elif action.action_type == "ignore" and hazard != "normal":
            # Ignoring a real hazard is bad
            return GrainStoreReward(
                value=0.0,
                reason=f"Dangerous! Ignoring {hazard} risks grain spoilage."
            )
        else:
            return GrainStoreReward(
                value=0.2,
                reason=f"Incorrect. Best action for {hazard} was {correct_action}."
            )

    # ── Task 3 Grader: Multi-hazard, partial credit per hazard ─
    def _grade_task3(self, action, hazards) -> GrainStoreReward:
        correct_action = self.CORRECT_ACTIONS.get(
            hazards[self._state.step_count - 1], "ignore"
        )
        hazard = hazards[self._state.step_count - 1]

        if action.action_type == correct_action:
            return GrainStoreReward(
                value=1.0,
                reason=f"Step {self._state.step_count}: Correct for {hazard}."
            )
        elif action.action_type in ["alert", "inspect"]:
            return GrainStoreReward(
                value=0.5,
                reason=f"Step {self._state.step_count}: Cautious but {correct_action} was better."
            )
        else:
            return GrainStoreReward(
                value=0.0,
                reason=f"Step {self._state.step_count}: Wrong action for {hazard}."
            )

    # ════════════════════════════════════════════════════════
    #  HELPERS
    # ════════════════════════════════════════════════════════
    def _build_observation(self, message: str) -> GrainStoreObservation:
        sensors = self._scenario.get("sensors", {
            "temperature": 25.0, "humidity": 50.0,
            "gas_level": 200.0,  "fill_level": 50.0,
            "vibration": False
        })

        descriptions = {
            "task1": "TASK 1 (Easy): Identify the hazard and respond with the correct action.",
            "task2": "TASK 2 (Medium): One sensor is critical. Choose the best intervention.",
            "task3": "TASK 3 (Hard): Multiple hazards active. Respond to each one correctly.",
        }

        return GrainStoreObservation(
            temperature=sensors["temperature"],
            humidity=sensors["humidity"],
            gas_level=sensors["gas_level"],
            fill_level=sensors["fill_level"],
            vibration=sensors["vibration"],
            task_id=self._task_id,
            task_description=descriptions.get(self._task_id, ""),
            step_count=self._state.step_count,
            message=message,
        )

    def _feedback_message(self, action, reward, done) -> str:
        msg = f"Action '{action.action_type}' on '{action.target_zone}' → score {reward.value:.1f}. {reward.reason}"
        if done:
            msg += f" | Episode complete. Total reward: {self._state.total_reward:.2f}"
        return msg