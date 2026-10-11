from __future__ import annotations

import numpy as np
import pandas as pd

from gait_analysis.mfpca import fit_mfpca, hierarchical_step_pair_weights, trajectory_tensor


def _known_modes_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(4)
    phase = np.linspace(0.0, 1.0, 101)
    mode1 = np.stack([(index + 1) * np.sin(2 * np.pi * phase) for index in range(6)])
    mode2 = np.stack([np.cos(2 * np.pi * phase + index / 6) for index in range(6)])
    a, b = rng.normal(size=(2, 80))
    data = a[:, None, None] * mode1 + 0.35 * b[:, None, None] * mode2 + rng.normal(scale=0.01, size=(80, 6, 101))
    return data, a, b


def test_known_modes_weighted_orthonormality_and_reconstruction() -> None:
    data, _, _ = _known_modes_dataset()
    result = fit_mfpca(data, target_fve=0.99)
    assert np.all(np.diff(result.eigenvalues) <= 0)
    assert result.n_components == 2
    metric = np.repeat(result.channel_weights / 101, 101)
    phi = result.eigenfunctions.reshape(len(result.eigenfunctions), -1)
    np.testing.assert_allclose((phi * metric) @ phi.T, np.eye(len(phi)), atol=1e-6)
    assert np.median(result.reconstruction_error) < 0.1


def test_n_components_none_matches_pre_override_target_fve_behavior() -> None:
    """Additive kwarg: the default ``n_components=None`` must reproduce the exact
    pre-change target_fve-driven component count and scores bit-for-bit."""
    data, _, _ = _known_modes_dataset()
    baseline = fit_mfpca(data, target_fve=0.99)
    explicit_default = fit_mfpca(data, target_fve=0.99, n_components=None)
    assert explicit_default.n_components == baseline.n_components == 2
    np.testing.assert_array_equal(explicit_default.scores, baseline.scores)
    np.testing.assert_array_equal(explicit_default.eigenfunctions, baseline.eigenfunctions)
    np.testing.assert_array_equal(explicit_default.eigenvalues, baseline.eigenvalues)
    np.testing.assert_array_equal(explicit_default.reconstruction_error, baseline.reconstruction_error)


def test_n_components_override_forces_fixed_rank_ignoring_target_fve() -> None:
    """A caller-supplied ``n_components`` must win over ``target_fve`` entirely."""
    data, _, _ = _known_modes_dataset()
    # target_fve=0.99 alone would select 2 components (per the baseline test above);
    # forcing n_components=1 must override that regardless.
    result = fit_mfpca(data, target_fve=0.99, n_components=1)
    assert result.n_components == 1
    assert result.scores.shape[1] >= result.n_components


def test_n_components_override_clamped_to_available_rank() -> None:
    """Requesting more components than exist must clamp rather than error."""
    data, _, _ = _known_modes_dataset()
    baseline = fit_mfpca(data, target_fve=0.999999)
    total_available = len(baseline.eigenvalues)
    result = fit_mfpca(data, n_components=total_available + 50)
    assert result.n_components == total_available


def test_single_channel_fit_with_n_components_override() -> None:
    """A genuine single-channel (node-level) fit, as used by the per-node FPCA pathway."""
    rng = np.random.default_rng(7)
    phase = np.linspace(0.0, 1.0, 51)
    mode1 = np.sin(2 * np.pi * phase)
    mode2 = np.cos(4 * np.pi * phase)
    a, b = rng.normal(size=(2, 30))
    data = (a[:, None] * mode1 + 0.4 * b[:, None] * mode2)[:, None, :] + rng.normal(scale=0.01, size=(30, 1, 51))
    result = fit_mfpca(data, channel_weighting="equal", n_components=2)
    assert result.n_components == 2
    assert result.eigenfunctions.shape == (len(result.eigenvalues), 1, 51)
    assert result.scores.shape[0] == 30


def test_hierarchical_weights_balance_participants_and_trials() -> None:
    metadata = pd.DataFrame({
        "participant_id": ["a", "a", "a", "b"], "trial_id": ["x", "x", "y", "z"],
        "recording_id": ["r"] * 4, "step_pair_id": ["1", "2", "3", "4"],
    })
    weights = hierarchical_step_pair_weights(metadata)
    np.testing.assert_allclose([weights[:3].sum(), weights[3:].sum()], [0.5, 0.5])


def test_trajectory_tensor_preserves_plain_identifier_columns() -> None:
    rows = []
    for participant, trial, recording, pair in (("p1", "t1", "r1", "p1"), ("p2", "t2", "r2", "p2")):
        for node in ("LH", "LK"):
            for point in (0.0, 100.0):
                rows.append({
                    "participant_id": participant, "trial_id": trial,
                    "recording_id": recording, "step_pair_id": pair,
                    "representation": "event_registered", "node": node,
                    "percent_step": point, "value": 1.0,
                })

    tensor, metadata = trajectory_tensor(pd.DataFrame(rows), ("LH", "LK"), "event_registered")

    assert tensor.shape == (2, 2, 2)
    assert metadata.columns.tolist() == ["participant_id", "trial_id", "recording_id", "step_pair_id"]
    assert next(metadata.itertuples(index=False)).participant_id == "p1"
    np.testing.assert_allclose(hierarchical_step_pair_weights(metadata), [0.5, 0.5])
