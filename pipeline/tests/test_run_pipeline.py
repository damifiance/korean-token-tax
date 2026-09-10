from klen.run_pipeline import resolve_estimator_local_path


def test_resolve_estimator_local_path_relative_to_spec(tmp_path):
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    spec_path = spec_dir / "measurement_spec.yaml"

    resolved = resolve_estimator_local_path(
        spec_path, "../models/Qwen3-1.7B-Base-ea980cb"
    )

    assert resolved == str(
        (tmp_path / "models/Qwen3-1.7B-Base-ea980cb").resolve()
    )


def test_resolve_estimator_local_path_preserves_absolute_and_none(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    absolute = (tmp_path / "snapshot").resolve()

    assert resolve_estimator_local_path(spec_path, str(absolute)) == str(absolute)
    assert resolve_estimator_local_path(spec_path, None) is None
