"""Repository invariants that protect the evidence from the surrounding tooling.

These are the Phase 0 hygiene gates: campaign scripts may only resolve the measured runtime through
scripts/_runtime.sh, the checkout declares its line endings and its Docker context, and the .env files no
longer advertise configuration keys that nothing reads.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_campaign_scripts_use_the_shared_runtime_resolver() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    assert scripts, "no campaign scripts found"
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "_runtime.sh" in text, f"{script.name} does not source the runtime resolver"
        assert ".venv-check/Scripts/python.exe" not in text, f"{script.name} still prefers the CI venv"
        assert 'PY=.venv' not in text, f"{script.name} hard-codes an interpreter instead of using the resolver"


def test_checkout_declares_line_endings_and_a_docker_context() -> None:
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "text=auto" in attributes and "eol=lf" in attributes
    dockerignore = (ROOT / "ctde_arena" / ".dockerignore").read_text(encoding="utf-8")
    for pattern in ("data", "results", ".venv", "__pycache__"):
        assert pattern in dockerignore, f".dockerignore does not exclude {pattern}"


def test_env_files_no_longer_advertise_dead_keys() -> None:
    for relative in (".env", ".env.example", "ctde_arena/.env", "ctde_arena/.env.example"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "PLOT_UPDATE_INTERVAL" not in text, relative
        assert "METRICS_FLUSH_INTERVAL" not in text, relative


def test_runtime_guard_is_wired_into_the_study_runner() -> None:
    source = (ROOT / "scripts" / "run_study.py").read_text(encoding="utf-8")
    assert "require_measured_runtime" in source
    assert (ROOT / "scripts" / "check_runtime.py").is_file()
