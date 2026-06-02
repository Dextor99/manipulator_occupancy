from main import run_pipeline


def main():
    result = run_pipeline(source="mock", config_dir="config", max_frames=20, visualize=False)
    decision = result.safety_decision
    print(f"objects={len(result.objects)} risk={decision.level.value} min_distance={decision.min_distance:.3f}")


if __name__ == "__main__":
    main()
