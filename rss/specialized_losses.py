"""Signal-family-specific losses used by adaptive RSS regularization.

The implementations in this module are promoted from
``notebooks/rss_validation_specialized_losses.ipynb``.  They accept either one
candidate with shape ``(3, N)`` or a particle batch with shape ``(P, 3, N)``
and preserve PyTorch autograd for gradient-based RSS optimizers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


PAIR_INDICES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
HARMONIC_SPECTRAL_OVERLAP = "harmonic_spectral_overlap"
FILTERED_MULTILAG_CROSS_CORRELATION = "filtered_multilag_cross_correlation"
CHIRP_TIME_FREQUENCY_OVERLAP = "chirp_time_frequency_overlap"
PIECEWISE_SPARSE_INNOVATION = "piecewise_sparse_innovation"
SPECIALIZED_LOSSES: tuple[str, ...] = (
    HARMONIC_SPECTRAL_OVERLAP,
    FILTERED_MULTILAG_CROSS_CORRELATION,
    CHIRP_TIME_FREQUENCY_OVERLAP,
    PIECEWISE_SPARSE_INNOVATION,
)


@dataclass(frozen=True)
class SpecializedLossConfig:
    """Numerical controls retained from the specialized-loss notebook."""

    bad_loss: float = 1e6
    variance_floor: float = 1e-12
    numerical_eps: float = 1e-12
    unwrap_for_loss: bool = True
    analysis_band_hz: tuple[float, float] = (0.0, 5.0)
    harmonic_nperseg: int = 256
    harmonic_overlap_fraction: float = 0.5
    harmonic_gamma: float = 1.0
    filtered_max_lag: int = 100
    chirp_nperseg: int = 128
    chirp_noverlap: int = 96
    chirp_nfft: int = 256
    chirp_energy_weighted: bool = False
    piecewise_lambda: float = 0.5

    def __post_init__(self) -> None:
        finite_positive = {
            "bad_loss": self.bad_loss,
            "variance_floor": self.variance_floor,
            "numerical_eps": self.numerical_eps,
            "harmonic_gamma": self.harmonic_gamma,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0.")
        low, high = map(float, self.analysis_band_hz)
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
            raise ValueError(
                "analysis_band_hz must contain finite increasing nonnegative bounds."
            )
        if int(self.harmonic_nperseg) < 4:
            raise ValueError("harmonic_nperseg must be >= 4.")
        if not 0.0 <= float(self.harmonic_overlap_fraction) < 1.0:
            raise ValueError("harmonic_overlap_fraction must lie in [0, 1).")
        if int(self.filtered_max_lag) < 0:
            raise ValueError("filtered_max_lag must be >= 0.")
        if int(self.chirp_nperseg) < 4:
            raise ValueError("chirp_nperseg must be >= 4.")
        if not 0 <= int(self.chirp_noverlap) < int(self.chirp_nperseg):
            raise ValueError("chirp_noverlap must lie in [0, chirp_nperseg).")
        if int(self.chirp_nfft) < int(self.chirp_nperseg):
            raise ValueError("chirp_nfft must be >= chirp_nperseg.")
        if not 0.0 <= float(self.piecewise_lambda) <= 1.0:
            raise ValueError("piecewise_lambda must lie in [0, 1].")


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(value - 1).bit_length())


def _unwrap_rows_torch(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] < 2:
        return values
    differences = values[..., 1:] - values[..., :-1]
    wrapped = torch.atan2(torch.sin(differences), torch.cos(differences))
    return torch.cat(
        (values[..., :1], values[..., :1] + torch.cumsum(wrapped, dim=-1)),
        dim=-1,
    )


def _pairwise_bhattacharyya(
    distributions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_values = torch.stack(
        [
            torch.sum(
                torch.sqrt(
                    distributions[..., first, :] * distributions[..., second, :]
                ),
                dim=-1,
            )
            for first, second in PAIR_INDICES
        ],
        dim=-1,
    )
    return torch.mean(pair_values, dim=-1), pair_values


def _one_sided_windowed_power(
    values: torch.Tensor,
    sample_rate_hz: float,
    nperseg: int,
    noverlap: int,
    nfft: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    sample_count = int(values.shape[-1])
    segment = min(int(nperseg), sample_count)
    if segment < 4:
        raise ValueError("Spectral losses require at least four samples.")
    overlap = min(int(noverlap), segment - 1)
    hop = segment - overlap
    fft_count = max(int(nfft), segment)
    frames = values.unfold(-1, segment, hop)
    window = torch.hann_window(
        segment,
        periodic=True,
        dtype=values.dtype,
        device=values.device,
    )
    transformed = torch.fft.rfft(frames * window, n=fft_count, dim=-1)
    power = transformed.real.square() + transformed.imag.square()
    scale = float(sample_rate_hz) * torch.sum(window.square())
    power = power / torch.clamp(scale, min=torch.finfo(values.dtype).tiny)
    scale_bins = torch.ones(
        int(power.shape[-1]), dtype=power.dtype, device=power.device
    )
    if fft_count % 2 == 0 and power.shape[-1] > 2:
        scale_bins[1:-1] = 2.0
    elif power.shape[-1] > 1:
        scale_bins[1:] = 2.0
    power = power * scale_bins
    frequencies = torch.fft.rfftfreq(
        fft_count, d=1.0 / float(sample_rate_hz), device=values.device
    )
    return power, frequencies, segment, overlap, hop, fft_count


def _harmonic_loss(
    values: torch.Tensor,
    sample_rate_hz: float,
    config: SpecializedLossConfig,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    nperseg = min(int(config.harmonic_nperseg), int(values.shape[-1]))
    noverlap = int(math.floor(float(config.harmonic_overlap_fraction) * nperseg))
    nfft = _next_power_of_two(nperseg)
    frame_power, frequencies, nperseg, noverlap, _, nfft = (
        _one_sided_windowed_power(
            values, sample_rate_hz, nperseg, noverlap, nfft
        )
    )
    psd = torch.mean(frame_power, dim=-2)
    low, high = config.analysis_band_hz
    mask = (frequencies >= float(low)) & (frequencies <= float(high))
    if not bool(torch.any(mask)):
        raise ValueError("The harmonic analysis band contains no FFT bins.")
    band_psd = psd[..., mask]
    shaped = torch.pow(
        band_psd + float(config.numerical_eps), float(config.harmonic_gamma)
    )
    distributions = shaped / torch.clamp(
        torch.sum(shaped, dim=-1, keepdim=True),
        min=float(config.numerical_eps),
    )
    loss, pairs = _pairwise_bhattacharyya(distributions)
    detail: dict[str, object] = {}
    if diagnostics:
        detail = {
            "frequencies_hz": frequencies[mask],
            "psd": band_psd,
            "spectral_distributions": distributions,
            "pair_overlaps": pairs,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
        }
    return loss, detail, torch.ones_like(loss, dtype=torch.bool)


def _filtered_loss(
    values: torch.Tensor,
    config: SpecializedLossConfig,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    sample_count = int(values.shape[-1])
    maximum_lag = min(
        int(config.filtered_max_lag), sample_count // 4, sample_count - 1
    )
    lags = torch.arange(
        -maximum_lag,
        maximum_lag + 1,
        device=values.device,
        dtype=torch.long,
    )
    lag_float = lags.to(dtype=values.dtype)
    weights = 1.0 - torch.abs(lag_float) / float(maximum_lag + 1)
    weights = weights / torch.sum(weights)
    convolution_nfft = _next_power_of_two(2 * sample_count - 1)
    spectra = torch.fft.rfft(values, n=convolution_nfft, dim=-1)
    reversed_spectra = torch.fft.rfft(
        torch.flip(values, dims=(-1,)), n=convolution_nfft, dim=-1
    )
    squares = values.square()
    prefix = torch.cat(
        (torch.zeros_like(squares[..., :1]), torch.cumsum(squares, dim=-1)),
        dim=-1,
    )
    x_start = torch.clamp(-lags, min=0)
    x_stop = sample_count - torch.clamp(lags, min=0)
    y_start = torch.clamp(lags, min=0)
    y_stop = sample_count - torch.clamp(-lags, min=0)
    correlations = []
    pair_energies = []
    for first, second in PAIR_INDICES:
        convolution = torch.fft.irfft(
            spectra[..., second, :] * reversed_spectra[..., first, :],
            n=convolution_nfft,
            dim=-1,
        )[..., : 2 * sample_count - 1]
        numerators = convolution[..., lags + sample_count - 1]
        first_prefix = prefix[..., first, :]
        second_prefix = prefix[..., second, :]
        first_energy = first_prefix[..., x_stop] - first_prefix[..., x_start]
        second_energy = second_prefix[..., y_stop] - second_prefix[..., y_start]
        denominator = torch.sqrt(
            torch.clamp(first_energy * second_energy, min=0.0)
            + float(config.numerical_eps)
        )
        rho = numerators / (denominator + float(config.numerical_eps))
        correlations.append(rho)
        pair_energies.append(torch.sum(weights * rho.square(), dim=-1))
    correlation_tensor = torch.stack(correlations, dim=-2)
    pair_energy_tensor = torch.stack(pair_energies, dim=-1)
    loss = torch.mean(pair_energy_tensor, dim=-1)
    detail = (
        {
            "lags": lags,
            "lag_weights": weights,
            "pair_correlations": correlation_tensor,
            "pair_energies": pair_energy_tensor,
        }
        if diagnostics
        else {}
    )
    return loss, detail, torch.ones_like(loss, dtype=torch.bool)


def _chirp_loss(
    values: torch.Tensor,
    sample_rate_hz: float,
    config: SpecializedLossConfig,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    frame_power, frequencies, nperseg, noverlap, hop, nfft = (
        _one_sided_windowed_power(
            values,
            sample_rate_hz,
            config.chirp_nperseg,
            config.chirp_noverlap,
            config.chirp_nfft,
        )
    )
    low, high = config.analysis_band_hz
    mask = (frequencies >= float(low)) & (frequencies <= float(high))
    if not bool(torch.any(mask)):
        raise ValueError("The chirp analysis band contains no FFT bins.")
    band_power = frame_power[..., mask]
    denominator = torch.sum(
        band_power + float(config.numerical_eps), dim=-1, keepdim=True
    )
    distributions = (band_power + float(config.numerical_eps)) / denominator
    frame_overlaps = torch.stack(
        [
            torch.sum(
                torch.sqrt(
                    distributions[..., first, :, :]
                    * distributions[..., second, :, :]
                ),
                dim=-1,
            )
            for first, second in PAIR_INDICES
        ],
        dim=-1,
    )
    if config.chirp_energy_weighted:
        total_energy = torch.sum(band_power, dim=(-3, -1))
        frame_weights = total_energy / torch.clamp(
            torch.sum(total_energy, dim=-1, keepdim=True),
            min=float(config.numerical_eps),
        )
    else:
        frame_weights = torch.full_like(
            frame_overlaps[..., 0], 1.0 / frame_overlaps.shape[-2]
        )
    pair_overlaps = torch.sum(
        frame_weights.unsqueeze(-1) * frame_overlaps, dim=-2
    )
    loss = torch.mean(pair_overlaps, dim=-1)
    detail: dict[str, object] = {}
    if diagnostics:
        band_frequencies = frequencies[mask]
        ridges = torch.sum(distributions * band_frequencies, dim=-1)
        frame_times = (
            torch.arange(
                frame_power.shape[-2], dtype=values.dtype, device=values.device
            )
            * hop
            + 0.5 * nperseg
        ) / float(sample_rate_hz)
        detail = {
            "frequencies_hz": band_frequencies,
            "frame_times_seconds": frame_times,
            "tf_power": band_power,
            "tf_distributions": distributions,
            "frame_overlaps": frame_overlaps,
            "pair_overlaps": pair_overlaps,
            "ridge_centroids_hz": ridges,
            "frame_weights": frame_weights,
            "nperseg": nperseg,
            "noverlap": noverlap,
            "nfft": nfft,
        }
    return loss, detail, torch.ones_like(loss, dtype=torch.bool)


def _piecewise_loss(
    values: torch.Tensor,
    config: SpecializedLossConfig,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    if values.shape[-1] < 3:
        raise ValueError("The piecewise loss requires at least three samples.")
    innovations = values[..., 2:] - 2.0 * values[..., 1:-1] + values[..., :-2]
    envelopes = torch.abs(innovations)
    count = int(innovations.shape[-1])
    l1 = torch.sum(torch.abs(innovations), dim=-1)
    l2 = torch.linalg.vector_norm(innovations, dim=-1)
    valid = torch.all(l2 > float(config.numerical_eps), dim=-1)
    root = math.sqrt(float(count))
    hoyer = (root - l1 / (l2 + float(config.numerical_eps))) / max(
        root - 1.0, config.numerical_eps
    )
    hoyer = torch.clamp(hoyer, min=0.0, max=1.0)
    sparse_loss = 1.0 - torch.mean(hoyer, dim=-1)
    event_overlaps = torch.stack(
        [
            torch.sum(
                envelopes[..., first, :] * envelopes[..., second, :], dim=-1
            )
            / (
                l2[..., first] * l2[..., second]
                + float(config.numerical_eps)
            )
            for first, second in PAIR_INDICES
        ],
        dim=-1,
    )
    event_loss = torch.mean(event_overlaps, dim=-1)
    weight = float(config.piecewise_lambda)
    loss = weight * sparse_loss + (1.0 - weight) * event_loss
    detail = (
        {
            "innovations": innovations,
            "event_envelopes": envelopes,
            "hoyer_sparsities": hoyer,
            "sparsity_loss": sparse_loss,
            "pair_event_overlaps": event_overlaps,
            "event_loss": event_loss,
        }
        if diagnostics
        else {}
    )
    return loss, detail, valid


def _specialized_loss_tensor(
    sources: torch.Tensor,
    sample_rate_hz: float,
    loss_type: str,
    config: SpecializedLossConfig,
    *,
    diagnostics: bool,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor]:
    values = sources
    if values.ndim not in (2, 3) or values.shape[-2] != 3:
        raise ValueError("sources must have shape (3, N) or (P, 3, N).")
    if values.shape[-1] < 4:
        raise ValueError("Specialized RSS losses require at least four samples.")
    if not math.isfinite(float(sample_rate_hz)) or float(sample_rate_hz) <= 0.0:
        raise ValueError("sample_rate_hz must be finite and > 0.")
    selected = str(loss_type).strip().lower()
    if selected not in SPECIALIZED_LOSSES:
        raise ValueError(
            f"loss_type must be one of: {', '.join(SPECIALIZED_LOSSES)}."
        )
    finite = torch.all(torch.isfinite(values), dim=(-2, -1))
    safe = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if config.unwrap_for_loss:
        safe = _unwrap_rows_torch(safe)
    safe = safe - torch.mean(safe, dim=-1, keepdim=True)
    variances = torch.mean(safe.square(), dim=-1)
    valid = finite & torch.all(variances >= float(config.variance_floor), dim=-1)
    if selected == HARMONIC_SPECTRAL_OVERLAP:
        loss, detail, extra_valid = _harmonic_loss(
            safe, sample_rate_hz, config, diagnostics
        )
    elif selected == FILTERED_MULTILAG_CROSS_CORRELATION:
        loss, detail, extra_valid = _filtered_loss(safe, config, diagnostics)
    elif selected == CHIRP_TIME_FREQUENCY_OVERLAP:
        loss, detail, extra_valid = _chirp_loss(
            safe, sample_rate_hz, config, diagnostics
        )
    else:
        loss, detail, extra_valid = _piecewise_loss(safe, config, diagnostics)
    valid = valid & extra_valid & torch.isfinite(loss)
    guarded = torch.where(
        valid, loss, torch.full_like(loss, float(config.bad_loss))
    )
    if diagnostics:
        detail = dict(detail)
        detail["centered_unwrapped_sources"] = safe
        detail["row_variances"] = variances
        detail["valid_candidate"] = valid
    return guarded, detail, valid


def _to_numpy_diagnostics(value: object) -> object:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return float(array) if array.ndim == 0 else array
    if isinstance(value, dict):
        return {key: _to_numpy_diagnostics(item) for key, item in value.items()}
    return value


def compute_specialized_loss(
    sources: np.ndarray | torch.Tensor,
    sample_rate_hz: float,
    loss_type: str,
    config: SpecializedLossConfig | None = None,
    *,
    return_diagnostics: bool = False,
) -> object:
    """Evaluate one promoted specialized loss.

    NumPy inputs produce Python/NumPy outputs.  Torch inputs stay on their
    original device and remain differentiable.
    """

    active_config = config or SpecializedLossConfig()
    is_tensor = isinstance(sources, torch.Tensor)
    tensor = (
        sources
        if is_tensor
        else torch.as_tensor(sources, dtype=torch.float64)
    )
    loss, diagnostics, _ = _specialized_loss_tensor(
        tensor,
        sample_rate_hz,
        loss_type,
        active_config,
        diagnostics=return_diagnostics,
    )
    if is_tensor:
        return (loss, diagnostics) if return_diagnostics else loss
    converted_loss = _to_numpy_diagnostics(loss)
    converted_diagnostics = _to_numpy_diagnostics(diagnostics)
    return (
        (converted_loss, converted_diagnostics)
        if return_diagnostics
        else converted_loss
    )


__all__ = [
    "CHIRP_TIME_FREQUENCY_OVERLAP",
    "FILTERED_MULTILAG_CROSS_CORRELATION",
    "HARMONIC_SPECTRAL_OVERLAP",
    "PIECEWISE_SPARSE_INNOVATION",
    "SPECIALIZED_LOSSES",
    "SpecializedLossConfig",
    "compute_specialized_loss",
]
