import requests


class Client:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def reset(self, difficulty="easy", seed=42):
        r = requests.post(
            f"{self.base_url}/reset",
            json={"difficulty": difficulty, "seed": seed},
        )
        r.raise_for_status()
        return r.json()

    def step(self, action: dict):
        r = requests.post(
            f"{self.base_url}/step",
            json={"action": action},
        )
        r.raise_for_status()
        return r.json()

    def state(self):
        r = requests.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()