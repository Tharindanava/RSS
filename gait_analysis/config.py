"""Validated YAML configuration for the gait-analysis toolkit."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .utils import hash_mapping


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


@dataclass(frozen=True)
class AnalysisConfig:
    """Typed container around the complete, hashable analysis configuration."""

    values: dict[str, Any]
    source_path: Path
    profile: str = "development"

    @classmethod
    def from_yaml(
        cls,
        path: str | Path | None = None,
        *,
        overlays: list[str | Path] | None = None,
        profile: str = "development",
    ) -> "AnalysisConfig":
        default = Path(__file__).resolve().parent / "configs" / "default.yaml"
        source = Path(path or default).resolve()
        values = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        for overlay_path in overlays or []:
            overlay = yaml.safe_load(Path(overlay_path).read_text(encoding="utf-8")) or {}
            values = _deep_merge(values, overlay)
        config = cls(values=dict(values), source_path=source, profile=profile)
        config.validate()
        return config

    @property
    def config_hash(self) -> str:
        return hash_mapping({"values": self.values, "profile": self.profile})

    @property
    def project_root(self) -> Path:
        raw = Path(str(self.values.get("project_root", "..")))
        if raw.is_absolute():
            return raw.resolve()
        configured = (self.source_path.parent / raw).resolve()
        if (configured / "Sample_Data").exists():
            return configured
        working_directory = Path.cwd().resolve()
        if (working_directory / "Sample_Data").exists():
            return working_directory
        return configured

    def path(self, name: str) -> Path:
        raw = Path(str(self.values["paths"][name]))
        return raw.resolve() if raw.is_absolute() else (self.project_root / raw).resolve()

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise KeyError(f"Configuration section '{name}' is missing")
        return value

    def with_values(self, overlay: Mapping[str, Any]) -> "AnalysisConfig":
        result = AnalysisConfig(_deep_merge(self.values, overlay), self.source_path, self.profile)
        result.validate()
        return result

    def validate(self) -> None:
        required = {
            "paths", "event_sensors", "proxy", "segmentation", "classic", "mfpca",
            "transfer_entropy", "phenotype", "clustering", "statistics", "profiles",
            "scenario_filter", "node_fpca",
        }
        missing = sorted(required.difference(self.values))
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
        if self.profile not in self.values["profiles"]:
            raise ValueError(f"Unknown run profile: {self.profile}")
        if str(self.values.get("manifest", {}).get("checksum_algorithm", "sha256")).lower() != "sha256":
            raise ValueError("Only the reproducibility-locked SHA-256 checksum algorithm is supported")
        segment = self.values["segmentation"]
        low, high = map(float, segment["bandpass_hz"])
        if not 0.0 < low < high:
            raise ValueError("segmentation.bandpass_hz must be increasing and positive")
        stride_low, stride_high = map(float, segment["stride_seconds"])
        if not 0.0 < stride_low < stride_high:
            raise ValueError("segmentation.stride_seconds is invalid")
        threshold = float(segment["hilbert_phase_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError("segmentation.hilbert_phase_threshold must be within (0, 1)")
        rotation_low, rotation_high = map(float, segment["phase_rotation_bounds"])
        if not 0.0 < rotation_low < rotation_high:
            raise ValueError("segmentation.phase_rotation_bounds is invalid")
        te = self.values["transfer_entropy"]
        if len(te["nodes"]) != 12 or int(te["bins"]) < 2:
            raise ValueError("Primary transfer-entropy configuration requires 12 nodes and at least two bins")
        if int(te["history_source"]) < 1 or int(te["history_target"]) < 1:
            raise ValueError("Transfer-entropy history orders must be positive")
        if sorted(map(int, self.values["clustering"]["candidate_clusters"])) != list(
            map(int, self.values["clustering"]["candidate_clusters"])
        ):
            raise ValueError("Candidate cluster counts must be sorted")
        dbscan = self.values["clustering"].get("dbscan")
        if not isinstance(dbscan, dict):
            raise ValueError("clustering.dbscan configuration is required")
        if int(dbscan["min_samples"]) < 1:
            raise ValueError("clustering.dbscan.min_samples must be at least 1")
        if int(dbscan["embedding_dimensions"]) < 2:
            raise ValueError("clustering.dbscan.embedding_dimensions must be at least 2")
        if int(dbscan["min_cluster_trials"]) < 1:
            raise ValueError("clustering.dbscan.min_cluster_trials must be at least 1")
        if not 0.0 < float(dbscan["min_cluster_fraction"]) < 1.0:
            raise ValueError("clustering.dbscan.min_cluster_fraction must be within (0, 1)")
        eps_percentiles = list(map(float, dbscan["eps_candidate_percentiles"]))
        if sorted(eps_percentiles) != eps_percentiles or not all(0.0 < p < 100.0 for p in eps_percentiles):
            raise ValueError("clustering.dbscan.eps_candidate_percentiles must be sorted values within (0, 100)")
        if not 0.0 <= float(dbscan["max_noise_fraction"]) < 1.0:
            raise ValueError("clustering.dbscan.max_noise_fraction must be within [0, 1)")
        scenario_filter = self.values["scenario_filter"]
        if int(scenario_filter["min_subjects_per_scenario"]) < 1:
            raise ValueError("scenario_filter.min_subjects_per_scenario must be at least 1")
        node_fpca = self.values["node_fpca"]
        if int(node_fpca["n_eigenfunctions"]) < 1:
            raise ValueError("node_fpca.n_eigenfunctions must be at least 1")
        valid_representations = {"timing_preserving", "event_registered"}
        if str(node_fpca["representation"]) not in valid_representations:
            raise ValueError(f"node_fpca.representation must be one of {sorted(valid_representations)}")
        node_fpca_nodes = set(map(str, node_fpca["nodes"]))
        for scope, nodes in self.values["clustering"].get("analyses", {}).items():
            invalid = sorted(set(map(str, nodes)) - node_fpca_nodes)
            if invalid:
                raise ValueError(
                    f"clustering.analyses[{scope!r}] references node(s) not in node_fpca.nodes: {invalid}"
                )
