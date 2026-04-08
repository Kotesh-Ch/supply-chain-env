import requests

URL = "https://kotesh-ch-supply-chain-env.hf.space/step"


def get_action(obs):
    inventory = obs["nodes"][0]["inventory"]
    incoming = sum(x["quantity"] for x in obs["in_transit"])
    demand = obs["nodes"][0]["demand_per_step"]

    lead_time = 2

    total_future = inventory + incoming

    # 🔥 FIXED LOGIC
    reorder_point = demand * (lead_time + 1)     # 24
    order_up_to = demand * (lead_time + 2)       # 32

    # 🚫 if already incoming → don't spam orders
    if incoming > 0:
        return {"type": "wait"}

    # ✅ order EARLY (not late)
    if total_future <= reorder_point:
        quantity = order_up_to - total_future

        return {
            "type": "order",
            "supplier_id": "S1",
            "quantity": int(quantity)
        }

    return {"type": "wait"}

def run():
    action = {"type": "wait"}  # initial action

    for step in range(15):
        res = requests.post(URL, json={"action": action}).json()

        obs = res["observation"]
        reward = res["reward"]

        inventory = obs["nodes"][0]["inventory"]
        incoming = obs["in_transit"]

        print(f"\nStep {step + 1}")
        print("Inventory:", inventory)
        print("Incoming:", incoming)
        print("Reward:", reward)

        # Decide next action
        action = get_action(obs)


if __name__ == "__main__":
    run()