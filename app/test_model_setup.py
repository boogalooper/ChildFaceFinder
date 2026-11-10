from __future__ import annotations

from pathlib import Path

import model_setup


def _make_valid_model(path: Path) -> None:
    path.mkdir(parents=True)
    payload = b"x" * 100_001
    for index in range(5):
        (path / f"model_{index}.onnx").write_bytes(payload)


def test_recovery_replaces_invalid_active_model_with_valid_backup(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "models"
    active = root / "antelopev2"
    active.mkdir(parents=True)
    (active / "broken.onnx").write_bytes(b"broken")
    backup = root / ".antelopev2.previous.test"
    _make_valid_model(backup)

    monkeypatch.setattr(model_setup, "MODEL_DIR", active)
    monkeypatch.setattr(model_setup, "MODEL_NAME", "antelopev2")

    model_setup._recover_interrupted_model_swap()

    assert model_setup.model_is_valid(active)
    assert not backup.exists()
    assert not list(root.glob(".antelopev2.damaged.*"))


def test_model_install_lock_prevents_removal_of_other_staging(tmp_path, monkeypatch):
    import pytest
    from runtime_safety import exclusive_lock
    active = tmp_path/'antelopev2'
    _make_valid_model(active)
    candidate = tmp_path/'.antelopev2.new.active'
    candidate.mkdir()
    monkeypatch.setattr(model_setup, 'MODEL_DIR', active)
    with exclusive_lock(tmp_path/'.antelopev2.install.lock'):
        with pytest.raises(RuntimeError, match='другая операция'):
            model_setup.ensure_antelopev2()
    assert candidate.exists()
    assert model_setup.model_is_valid(active)
