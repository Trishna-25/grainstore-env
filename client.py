import httpx
import asyncio
from models import GrainStoreAction

class GrainStoreEnv:
    """
    Client to connect to the GrainStore-Env server.
    Used by inference.py and training scripts.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)

    # ── Connect ───────────────────────────────────────────
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.client.aclose()

    # ── Reset ─────────────────────────────────────────────
    async def reset(self, task_id: str = "task1"):
        """Start a new episode."""
        response = await self.client.post(
            f"{self.base_url}/reset",
            params={"task_id": task_id}
        )
        return response.json()

    # ── Step ──────────────────────────────────────────────
    async def step(self, action: GrainStoreAction):
        """Send action to environment."""
        response = await self.client.post(
            f"{self.base_url}/step",
            json=action.model_dump()
        )
        return response.json()

    # ── State ─────────────────────────────────────────────
    async def state(self):
        """Get current environment state."""
        response = await self.client.get(f"{self.base_url}/state")
        return response.json()

    # ── Health ────────────────────────────────────────────
    async def health(self):
        """Check if server is alive."""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()

    # ── Sync wrapper ──────────────────────────────────────
    @classmethod
    def from_docker_image(cls, base_url: str = "http://localhost:7860"):
        """Create client pointing to a running server."""
        return cls(base_url=base_url)


# ── Quick test ────────────────────────────────────────────
async def test():
    async with GrainStoreEnv() as env:
        print("=== Testing GrainStore-Env ===")

        # Test reset
        result = await env.reset(task_id="task1")
        print(f"\n[RESET] {result['observation']['message']}")
        print(f"Temperature: {result['observation']['temperature']}°C")
        print(f"Humidity: {result['observation']['humidity']}%")
        print(f"Gas Level: {result['observation']['gas_level']} ppm")
        print(f"Task: {result['observation']['task_description']}")

        # Test step
        action = GrainStoreAction(
            action_type="ventilate",
            target_zone="zone_A",
            reasoning="Temperature seems high"
        )
        result = await env.step(action)
        print(f"\n[STEP] Reward: {result['reward']['value']}")
        print(f"Reason: {result['reward']['reason']}")
        print(f"Done: {result['done']}")

        # Test state
        state = await env.state()
        print(f"\n[STATE] Total reward so far: {state['total_reward']}")


if __name__ == "__main__":
    asyncio.run(test())