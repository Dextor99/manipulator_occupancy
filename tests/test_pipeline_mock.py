from main import run_pipeline


def test_mock_pipeline_returns_objects_and_safety_decision():
    result = run_pipeline(source="mock", config_dir="config", max_frames=3, visualize=False)

    assert result.frames == 3
    assert len(result.objects) >= 1
    assert result.safety_decision.level.value in {"SAFE", "WARNING", "SLOW", "STOP"}
