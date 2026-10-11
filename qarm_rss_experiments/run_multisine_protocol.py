import csv
import json
import math
import time
from pathlib import Path
import numpy as np

try:
    from quanser.hardware import HIL
except ImportError as error:
    raise RuntimeError("Quanser Python API is required.") from error

# --- Configuration ---
# Update these variables to run specific batches (e.g., 1 to 5, then 6 to 10)
START_TEST = 31
END_TEST = 35

SAMPLE_RATE_HZ = 100.0
TEST_DURATION_SECONDS = 30.0
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# These are the same practical motion-rate and source-independence constraints used
# when preparing the experiment.  The component frequencies below remain in the
# original 0.05--0.12 Hz range; this limit additionally protects against a sum of
# components (or the decorrelation step) producing a faster target trajectory.
MAX_ANGULAR_VELOCITY_DEG_S = 45.0
SOURCE_CROSS_CORRELATION_LIMIT = 0.01
MIN_COMPONENT_FREQUENCY_HZ = 0.05
MAX_COMPONENT_FREQUENCY_HZ = 0.12

# Keep a margin below 5 degrees so tracking/rounding cannot put the commanded elbow
# at the geometric limit where the base and wrist axes cease to be approximately
# orthogonal.
ELBOW_MAX_ABS_DEG = 5.0
ELBOW_DESIGN_AMPLITUDE_DEG = 4.5

POSITION_CHANNELS = np.array([1000, 1001, 1002, 1003, 1004], dtype=np.uint32)
NUM_CHANNELS = len(POSITION_CHANNELS)
JOINT_NAMES = ["base", "shoulder", "elbow", "wrist", "gripper"]

HOME_POSE_DEG = np.array([0.0, 0.0, 0.0, 0.0, 45.0], dtype=np.float64)
MAX_AMPLITUDES_DEG = np.array(
    [60.0, 0.0, ELBOW_DESIGN_AMPLITUDE_DEG, 30.0, 0.0],
    dtype=np.float64,
)


def _cross_correlation_matrix(signals: np.ndarray) -> np.ndarray:
    """Return the mean-centered Pearson correlation of row-wise signals."""
    centered = signals - np.mean(signals, axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("Every active joint signal must have non-zero variance.")
    correlation = (centered @ centered.T) / (norms[:, None] * norms[None, :])
    return np.clip(correlation, -1.0, 1.0)


def _decorrelate_signals(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-center and symmetrically whiten row-wise source trajectories.

    This follows the simulation-signal construction used by ``rss_validation``.
    Symmetric whitening is preferred to sequential Gram-Schmidt here because it
    does not privilege any one QArm joint.
    """
    centered = signals - np.mean(signals, axis=1, keepdims=True)
    covariance = (centered @ centered.T) / centered.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    rank_tolerance = max(
        1e-12,
        float(np.max(eigenvalues))
        * np.finfo(np.float64).eps
        * max(covariance.shape),
    )
    if np.any(eigenvalues <= rank_tolerance):
        raise RuntimeError("Generated joint signals are not linearly independent.")

    whitening_matrix = (
        eigenvectors
        @ np.diag(np.reciprocal(np.sqrt(eigenvalues)))
        @ eigenvectors.T
    )
    decorrelated = whitening_matrix @ centered
    return decorrelated, whitening_matrix


def generate_trajectory(run_index: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate repeatable, mutually uncorrelated, rate-limited trajectories."""
    # The seed relies directly on run_index, ensuring Test X is identical across any batch size
    rng = np.random.RandomState(seed=20000 + run_index)
    t = np.linspace(
        0.0,
        TEST_DURATION_SECONDS,
        int(TEST_DURATION_SECONDS * SAMPLE_RATE_HZ),
        dtype=np.float64,
    )
    trajectory_deg = np.tile(HOME_POSE_DEG, (len(t), 1))
    active_indices = np.flatnonzero(MAX_AMPLITUDES_DEG > 0.0)
    raw_signals = np.empty((len(active_indices), len(t)), dtype=np.float64)
    generated_wave_info = {}

    # Initialize detailed metadata dictionary for this run
    run_metadata = {
        "test_number": run_index,
        "duration_seconds": float(TEST_DURATION_SECONDS),
        "sample_rate_hz": float(SAMPLE_RATE_HZ),
        "frequency_range_hz": [
            float(MIN_COMPONENT_FREQUENCY_HZ),
            float(MAX_COMPONENT_FREQUENCY_HZ),
        ],
        "maximum_angular_velocity_deg_s": float(MAX_ANGULAR_VELOCITY_DEG_S),
        "source_cross_correlation_limit": float(SOURCE_CROSS_CORRELATION_LIMIT),
        "elbow_max_abs_deg": float(ELBOW_MAX_ABS_DEG),
        "joints": {},
    }

    for i in range(NUM_CHANNELS):
        joint_name = JOINT_NAMES[i]

        # Log locked joints
        if MAX_AMPLITUDES_DEG[i] == 0.0:
            run_metadata["joints"][joint_name] = {
                "status": "locked",
                "home_pose_deg": float(HOME_POSE_DEG[i]),
            }

    # First construct one independent random multisine candidate per moving joint.
    # Whitening must be performed jointly, after all candidates have been created.
    for active_row, joint_index in enumerate(active_indices):
        signal = np.zeros_like(t)
        num_sines = int(rng.randint(2, 6))
        waves_info = []

        for _ in range(num_sines):
            freq = float(
                rng.uniform(MIN_COMPONENT_FREQUENCY_HZ, MAX_COMPONENT_FREQUENCY_HZ)
            )
            phase = float(rng.uniform(0, 2 * np.pi))
            waves_info.append({
                "frequency_hz": freq,
                "phase_rad": phase,
            })
            signal += np.sin(2 * np.pi * freq * t + phase)

        raw_signals[active_row] = signal
        generated_wave_info[joint_index] = {
            "status": "active",
            "num_sines": num_sines,
            # Retained for compatibility with existing result consumers.
            "target_amplitude_deg": float(MAX_AMPLITUDES_DEG[joint_index]),
            "requested_amplitude_deg": float(MAX_AMPLITUDES_DEG[joint_index]),
            "home_offset_deg": float(HOME_POSE_DEG[joint_index]),
            "raw_peak_before_centering": float(np.max(np.abs(signal))),
            "waves": waves_info,
        }

    decorrelated, whitening_matrix = _decorrelate_signals(raw_signals)

    # Scale rows independently. Positive row scaling preserves the zero correlation
    # established above. If necessary, reduce an amplitude to ensure the commanded
    # finite-difference velocity remains within the existing 45 deg/s profile rate.
    for active_row, joint_index in enumerate(active_indices):
        signal = decorrelated[active_row]
        decorrelated_peak = float(np.max(np.abs(signal)))
        signal_deg = signal * (
            MAX_AMPLITUDES_DEG[joint_index] / decorrelated_peak
        )
        peak_velocity = float(
            np.max(np.abs(np.diff(signal_deg) / np.diff(t)))
        )
        velocity_scale_factor = 1.0
        if peak_velocity > MAX_ANGULAR_VELOCITY_DEG_S:
            velocity_scale_factor = MAX_ANGULAR_VELOCITY_DEG_S / peak_velocity
            signal_deg *= velocity_scale_factor
            peak_velocity = float(
                np.max(np.abs(np.diff(signal_deg) / np.diff(t)))
            )

        trajectory_deg[:, joint_index] = (
            signal_deg + HOME_POSE_DEG[joint_index]
        )
        info = generated_wave_info[joint_index]
        info.update({
            "normalization_peak": decorrelated_peak,
            "velocity_scale_factor": float(velocity_scale_factor),
            "achieved_peak_amplitude_deg": float(np.max(np.abs(signal_deg))),
            "peak_angular_velocity_deg_s": peak_velocity,
            "decorrelation_coefficients": whitening_matrix[active_row].tolist(),
        })
        run_metadata["joints"][JOINT_NAMES[joint_index]] = info

    active_trajectories = trajectory_deg[:, active_indices].T
    correlation = _cross_correlation_matrix(active_trajectories)
    pairwise = np.abs(correlation[np.triu_indices(len(active_indices), k=1)])
    maximum_correlation = float(np.max(pairwise))
    if maximum_correlation >= SOURCE_CROSS_CORRELATION_LIMIT:
        raise RuntimeError(
            "Could not satisfy the source cross-correlation limit: "
            f"{maximum_correlation:.6g} >= {SOURCE_CROSS_CORRELATION_LIMIT:.6g}."
        )

    elbow_index = JOINT_NAMES.index("elbow")
    elbow_max_abs = float(np.max(np.abs(trajectory_deg[:, elbow_index])))
    if elbow_max_abs >= ELBOW_MAX_ABS_DEG:
        raise RuntimeError(
            f"Elbow trajectory reached {elbow_max_abs:.6g} deg; it must stay below "
            f"{ELBOW_MAX_ABS_DEG:.6g} deg."
        )

    active_velocities_deg_s = np.diff(
        trajectory_deg[:, active_indices], axis=0
    ) / np.diff(t)[:, None]
    maximum_velocity = float(np.max(np.abs(active_velocities_deg_s)))
    if maximum_velocity > MAX_ANGULAR_VELOCITY_DEG_S + 1e-9:
        raise RuntimeError(
            "Generated trajectory exceeds the angular-velocity limit: "
            f"{maximum_velocity:.6g} > {MAX_ANGULAR_VELOCITY_DEG_S:.6g} deg/s."
        )

    run_metadata["decorrelation"] = {
        "method": "symmetric_whitening",
        "active_joints": [JOINT_NAMES[index] for index in active_indices],
        "whitening_matrix": whitening_matrix.tolist(),
        "cross_correlation_matrix": correlation.tolist(),
        "maximum_absolute_pairwise_correlation": maximum_correlation,
    }
    run_metadata["elbow_achieved_max_abs_deg"] = elbow_max_abs
    run_metadata["achieved_max_angular_velocity_deg_s"] = maximum_velocity

    return t, np.radians(trajectory_deg), run_metadata


def command_pose_and_wait(card, target_rad: np.ndarray, timeout: float = 10.0):
    """Commands a static pose and waits for the arm to reach it."""
    card.write_other(POSITION_CHANNELS, NUM_CHANNELS, target_rad)
    start_time = time.perf_counter()

    while True:
        measured = np.empty(NUM_CHANNELS, dtype=np.float64)
        card.read_other(POSITION_CHANNELS, NUM_CHANNELS, measured)
        if float(np.max(np.abs(measured - target_rad))) <= math.radians(1.5):
            return
        if time.perf_counter() - start_time >= timeout:
            print("Warning: Did not fully settle within timeout.")
            return
        time.sleep(0.05)


def run_experiment_loop():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to QArm...")
    card = HIL()
    card.open("qarm_usb", "0")
    if not card.is_valid():
        raise RuntimeError("Failed to open QArm HIL card.")

    options = []
    for j in (0, 1, 2, 3):
        options.extend([
            f"j{j}_mode=0",
            f"j{j}_profile_config=0",
            f"j{j}_profile_velocity={math.radians(MAX_ANGULAR_VELOCITY_DEG_S):.6f}",
            f"j{j}_profile_acceleration=1.5",
        ])
    card.set_card_specific_options(",".join(options), 8192)

    try:
        home_rad = np.radians(HOME_POSE_DEG)
        print("Moving to initial Home position (Gripper will close)...")
        command_pose_and_wait(card, home_rad)

        # Loop relies on START_TEST and END_TEST configurations
        for run_idx in range(START_TEST, END_TEST + 1):
            print(f"\n--- Starting Test {run_idx} (Batch range: {START_TEST}-{END_TEST}) ---")
            t_array, targets_rad, metadata = generate_trajectory(run_idx)

            # 1. Print detailed wave logs to the terminal
            print("Wave Combinations Generated:")
            for j_name, data in metadata["joints"].items():
                if data["status"] == "active":
                    print(f"  - {j_name.capitalize()}: {data['num_sines']} waves")
                    for w_idx, wave in enumerate(data["waves"]):
                        print(f"      Wave {w_idx+1}: freq = {wave['frequency_hz']:.4f} Hz, phase = {wave['phase_rad']:.4f} rad")
            print(
                "Validated targets: max |correlation| = "
                f"{metadata['decorrelation']['maximum_absolute_pairwise_correlation']:.3e}, "
                f"max |elbow| = {metadata['elbow_achieved_max_abs_deg']:.3f} deg, "
                "max angular velocity = "
                f"{metadata['achieved_max_angular_velocity_deg_s']:.3f} deg/s"
            )

            # 2. Save exact wave configurations to test_X.json
            json_path = RESULTS_DIR / f"test_{run_idx}.json"
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump(metadata, json_file, indent=4)
            print(f"Saved: {json_path.name}")

            # 3. Execute motion and save CSV
            csv_path = RESULTS_DIR / f"test_{run_idx}.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                header = ["time_s"]
                for joint in JOINT_NAMES:
                    header.extend([f"target_{joint}_rad", f"measured_{joint}_rad"])
                writer.writerow(header)

                start_time = time.perf_counter()
                for i in range(len(t_array)):
                    scheduled_time = t_array[i]

                    while (time.perf_counter() - start_time) < scheduled_time:
                        pass

                    target = targets_rad[i]
                    card.write_other(POSITION_CHANNELS, NUM_CHANNELS, target)

                    position_buffer = np.empty(NUM_CHANNELS, dtype=np.float64)
                    card.read_other(POSITION_CHANNELS, NUM_CHANNELS, position_buffer)

                    row = [scheduled_time]
                    for j in range(NUM_CHANNELS):
                        row.extend([target[j], position_buffer[j]])
                    writer.writerow(row)

            print(f"Saved: {csv_path.name}")
            print("Returning to Home position for next run...")
            command_pose_and_wait(card, home_rad)

            # ---> MODIFICATION HERE <---
            print("Holding steady at Home position for 5 seconds...")
            time.sleep(5.0)

    except KeyboardInterrupt:
        print("\nExperiment manually interrupted.")
    finally:
        print("Securing arm and closing connection...")
        card.close()

if __name__ == "__main__":
    run_experiment_loop()
