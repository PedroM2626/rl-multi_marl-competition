from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List

from marl_arena.config import EXPORTS_DIR, METRICS_DIR
from marl_arena.models import MatchResult, TeamMetrics
from marl_arena.systems.plotting import export_metric_dashboard


class MetricsStore:
    def __init__(self, metrics_dir: Path | None = None, exports_dir: Path | None = None) -> None:
        self.metrics_dir = METRICS_DIR if metrics_dir is None else metrics_dir
        self.exports_dir = EXPORTS_DIR if exports_dir is None else exports_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.team_metrics_csv = self.metrics_dir / "team_match_metrics.csv"
        self.agent_metrics_csv = self.metrics_dir / "agent_match_metrics.csv"
        self.trajectory_metrics_csv = self.metrics_dir / "trajectory_metrics.csv"
        self.summary_json = self.metrics_dir / "summary.json"

    def _read_csv_header(self, file_path: Path) -> list[str] | None:
        if not file_path.exists():
            return None
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, None)

    def _rotate_csv_if_schema_changed(self, file_path: Path, fieldnames: list[str]) -> bool:
        existing_header = self._read_csv_header(file_path)
        if existing_header is None:
            return False
        if list(existing_header) == fieldnames:
            return False
        backup_path = file_path.with_name(f"{file_path.stem}.legacy{file_path.suffix}")
        counter = 0
        while backup_path.exists():
            counter += 1
            backup_path = file_path.with_name(f"{file_path.stem}.legacy{counter}{file_path.suffix}")
        file_path.rename(backup_path)
        warnings.warn(
            f"Esquema CSV alterado em {file_path.name}; arquivo anterior movido para {backup_path.name}."
        )
        return True

    def _append_rows(self, file_path: Path, rows: List[Dict[str, float]]) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        self._rotate_csv_if_schema_changed(file_path, fieldnames)
        file_exists = file_path.exists()
        with file_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def record_match(self, result: MatchResult, cumulative_metrics: Dict[str, TeamMetrics]) -> list[Path]:
        self._append_rows(self.team_metrics_csv, result.team_rows)
        self._append_rows(self.agent_metrics_csv, result.agent_rows)
        self._append_rows(self.trajectory_metrics_csv, result.trajectory_rows)
        self.write_summary(cumulative_metrics)
        try:
            return export_metric_dashboard(self.team_metrics_csv, self.exports_dir)
        except (ValueError, KeyError) as exc:
            warnings.warn(f"Nao foi possivel gerar dashboard de metricas: {exc}")
            return []

    def write_summary(self, cumulative_metrics: Dict[str, TeamMetrics]) -> None:
        payload = {
            "teams": [team_metrics.as_summary() for team_metrics in cumulative_metrics.values()],
        }
        with self.summary_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def latest_export_paths(self) -> Iterable[Path]:
        if not self.exports_dir.exists():
            return []
        return sorted(self.exports_dir.glob("*.png"))
