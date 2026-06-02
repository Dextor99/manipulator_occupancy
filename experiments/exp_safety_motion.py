from main import run_pipeline


def main():
    result = run_pipeline(source="mock", config_dir="config", max_frames=50, visualize=False)
    print(f"frames={result.frames} final_risk={result.safety_decision.level.value}")


if __name__ == "__main__":
    main()
