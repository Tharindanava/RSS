from __future__ import annotations

from pathlib import Path

import pandas as pd

import gait_analysis.pipeline as pipeline
from gait_analysis.models import TrialRecord
from gait_analysis.phenotype import build_trial_phenotypes
from gait_analysis.utils import hash_mapping


def test_missing_classic_identifiers_returns_empty_phenotypes(config) -> None:
    """An unavailable upstream family must not become a pandas merge-on-empty crash."""
    classic = pd.DataFrame(columns=["config_hash", "manifest_hash"])
    mfpca = pd.DataFrame(columns=["trial_id", "participant_id", "mfpca_score_1"])
    te = pd.DataFrame(columns=["trial_id", "te_density"])

    result = build_trial_phenotypes(classic, mfpca, te, config)

    assert set(result) == {"absolute", "terrain_adaptation", "footwear_adaptation"}
    assert all(frame.empty for frame in result.values())


def test_reference_aware_cache_versions_invalidate_all_legacy_stages(config) -> None:
    record = TrialRecord(
        "recording", "trial", "participant", Path("raw.csv"), Path("rss.csv"),
        raw_checksum="raw-hash", rss_checksum="rss-hash",
    )

    def legacy_key(stage: str) -> str:
        return hash_mapping({
            "stage": stage,
            "recording_id": record.recording_id,
            "raw": record.raw_checksum,
            "rss": record.rss_checksum,
            "config": config.config_hash,
        })

    assert pipeline._cache_key(record, config, "segment") != legacy_key("segment")
    assert pipeline._cache_key(record, config, "stage1") != legacy_key("stage1")
    assert pipeline._cache_key(record, config, "stage2") != legacy_key("stage2")
