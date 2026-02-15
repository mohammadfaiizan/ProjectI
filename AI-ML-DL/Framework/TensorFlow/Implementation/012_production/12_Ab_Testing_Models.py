"""
A/B testing, traffic splitting, canary deployment.
"""
import json
import os

def main():
    print("=" * 50)
    print("A/B Testing and Traffic Splitting")
    print("=" * 50)

    config = {
        "model_a": {"version": "1", "traffic": 0.9},
        "model_b": {"version": "2", "traffic": 0.1}
    }
    print("Traffic split config:", json.dumps(config, indent=2))

    import random
    def route_request():
        r = random.random()
        if r < config["model_a"]["traffic"]:
            return "model_a"
        return "model_b"

    results = {"model_a": 0, "model_b": 0}
    for _ in range(1000):
        results[route_request()] += 1
    print(f"Simulated 1000 requests: A={results['model_a']}, B={results['model_b']}")

    canary_config = {
        "stages": [
            {"version": "1", "traffic": 1.0},
            {"version": "2", "traffic": 0.05},
            {"version": "2", "traffic": 0.5},
            {"version": "2", "traffic": 1.0}
        ]
    }
    print("\nCanary deployment stages:")
    for i, s in enumerate(canary_config["stages"]):
        print(f"  Stage {i+1}: version {s['version']} at {s['traffic']*100}%")

    config_path = os.path.join(os.path.dirname(__file__), "ab_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    print("\nA/B testing concepts:")
    print("  - TF Serving: multiple model versions, version labels")
    print("  - Load balancer: route by header or random")
    print("  - Canary: gradual rollout with monitoring")

    print("A/B testing demo complete.")

if __name__ == "__main__":
    main()
