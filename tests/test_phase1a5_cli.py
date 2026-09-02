from pathlib import Path

from ppmoe.experiments.phase1a5_config import Phase1A5Config


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "phase1-v0_5.toml"


def test_cli_uses_explicit_artifact_root_and_preserves_checkpoints_on_registered_error(
    monkeypatch, tmp_path: Path
) -> None:
    from scripts import run_phase1a5

    config = Phase1A5Config.load(CONFIG_PATH)
    checkpoint = tmp_path / "checkpoints" / "local" / "signature"
    checkpoint.mkdir(parents=True)
    marker = checkpoint / "state.json"
    marker.write_text("keep", encoding="utf-8")
    called: dict[str, object] = {}

    monkeypatch.setattr(run_phase1a5.Phase1A5Config, "load", lambda _path: config)

    def failed(*_args, **kwargs):
        called.update(kwargs)
        raise run_phase1a5.StageError("registered gate")

    monkeypatch.setattr(run_phase1a5, "run_phase1a5", failed)

    assert (
        run_phase1a5.main(["--stage", "local", "--artifact-root", str(tmp_path)]) == 1
    )
    assert called["artifact_root"] == tmp_path
    assert marker.read_text(encoding="utf-8") == "keep"
