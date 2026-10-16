"""Committed artefacts must spell paths the way every host can read them.

Why this file exists: `scripts/run_absolute_eval.py` recorded the two files its derived blocks came from as
`str(Path)`, which on Windows is `results\\baseline\\random_baseline_uniform.json`. `--report-only` reads
that string back and joins it onto the repository root, so on a POSIX host it names one file whose name
contains backslashes -- and the artefact gate in CI crashed re-deriving its own provenance while passing on
the machine that wrote it ([A-42](../docs/11_code_audit.md)). The invariant is not "the numbers recompute",
which is platform-neutral, but "the record re-reads", which is not; only a check spanning every committed
JSON holds that.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Kept byte-identical on purpose. These are the archived per-seed summaries that document the pooled
# round-robin analysis which dropped a replication column from its match key (A-39); they are evidence of
# what that run recorded, not an instrument anyone re-reads, so rewriting their path spelling would falsify
# an artefact to satisfy a lint.
SUPERSEDED = {f"results/cross_arena/seeds/summary_seed{seed}.json" for seed in (1, 2, 3, 4, 5)}

# The keys whose values are repository paths, across the artefacts that are read back or cited.
# `source_scratch` is deliberately absent: it names the training run a checkpoint was copied out of, which
# is git-ignored by design, so requiring it would make a clean clone fail this gate.
PATH_KEYS = {"clip", "null_json", "compare_pool", "requirements", "manifest"}


def tracked_json() -> list[str]:
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout
    return sorted(p for p in listing.split("\0") if p.endswith(".json"))


def walk(payload: object, where: str = "") -> list[tuple[str, str]]:
    """(json pointer, value) for every string in the record, with the key it sits under in the pointer."""
    found: list[tuple[str, str]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                found.append((f"{where}/{key}", value))
            else:
                found += walk(value, f"{where}/{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if isinstance(value, str):
                found.append((f"{where}[{index}]", value))
            else:
                found += walk(value, f"{where}[{index}]")
    return found


def test_no_live_artefact_spells_a_path_with_a_host_separator() -> None:
    offenders = []
    for path in tracked_json():
        if path in SUPERSEDED:
            continue
        try:
            payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:  # a record this gate cannot read is a failure, not a skip
            offenders.append(f"{path}: unparseable ({error})")
            continue
        offenders += [f"{path}{pointer}: {value!r}"
                      for pointer, value in walk(payload) if "\\" in value]
    assert not offenders, (
        "a committed JSON records a path with a Windows separator, so a reader on a POSIX host cannot "
        "re-derive or re-open what the record points at. Serialise with `.as_posix()` and regenerate the "
        "artefact:\n  " + "\n  ".join(offenders)
    )


def test_every_path_a_committed_artefact_names_exists_on_this_host() -> None:
    """A provenance field that points at nothing is decoration, not provenance.

    A clone of this repository is the test environment: every path below is either versioned or derived by
    a command the documentation gives, which is what a reader has to re-open the record and check it.
    """
    missing = []
    for path in tracked_json():
        if path in SUPERSEDED:
            continue
        payload = json.loads((ROOT / path).read_text(encoding="utf-8"))
        for pointer, value in walk(payload):
            key = pointer.rsplit("/", 1)[-1]
            if key not in PATH_KEYS or not value or value.startswith(("http://", "https://")):
                continue
            if not (ROOT / value).exists():
                missing.append(f"{path}{pointer} -> {value}")
    assert not missing, "an artefact names a path that is not there:\n  " + "\n  ".join(missing)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_recorder_emits_forward_slashes_for_relative_and_absolute_paths() -> None:
    absolute_eval = _load("abs_eval_for_paths", ROOT / "scripts" / "run_absolute_eval.py")
    assert absolute_eval.recorded_path(None) is None
    assert absolute_eval.recorded_path(Path("results/baseline/random_baseline_uniform.json")) == (
        "results/baseline/random_baseline_uniform.json")
    assert absolute_eval.recorded_path(ROOT / "results" / "cross_arena" / "blocks6.json") == (
        "results/cross_arena/blocks6.json")
    # The defect was a host separator reaching the record; pin that the recorder cannot produce one.
    assert "\\" not in absolute_eval.recorded_path(Path("results") / "cross_arena" / "blocks.json")


def test_the_round_robin_harness_records_the_checkpoints_it_played_posix_relatively() -> None:
    """`summary.json`'s `policies` block is what makes a published table attributable to weights."""
    harness = _load("cross_arena_for_paths", ROOT / "scripts" / "run_cross_arena.py")
    policies = harness.replicate_policies(1, 7)
    recorded = {name: path.relative_to(harness.ROOT_ROOT).as_posix() for name, path in policies.items()}
    assert all("\\" not in value for value in recorded.values()), recorded
    for name, value in recorded.items():
        assert (ROOT / value).is_file(), f"{name} recorded as {value}, which is not in the repository"
