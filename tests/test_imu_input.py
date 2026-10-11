from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rss.imu_input import load_and_validate_input
from rss.sensor_layout import required_imu_columns


def _write_imu_csv(path: Path, *, device_prefix: str) -> None:
    data = {
        column.replace("device", device_prefix, 1): [float(index)]
        for index, column in enumerate(required_imu_columns())
    }
    pd.DataFrame(data).to_csv(path, index=False)


class InputLoadingTests(unittest.TestCase):
    def test_numeric_timestamp_is_treated_as_sample_counter(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "numeric_timestamp.csv"
            data = {
                column: [float(index)] * 4
                for index, column in enumerate(required_imu_columns())
            }
            data["Timestamp"] = [1, 2, 3, 4]
            pd.DataFrame(data).to_csv(csv_path, index=False)

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=True,
            )

            np.testing.assert_allclose(loaded.time_seconds, [0.0, 0.01, 0.02, 0.03])

    def test_datetime_timestamp_preserves_elapsed_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "datetime_timestamp.csv"
            data = {
                column: [float(index)] * 4
                for index, column in enumerate(required_imu_columns())
            }
            data["Timestamp"] = pd.date_range("2026-01-01", periods=4, freq="10ms")
            pd.DataFrame(data).to_csv(csv_path, index=False)

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=50.0,
                include_time_seconds=True,
            )

            np.testing.assert_allclose(loaded.time_seconds, [0.0, 0.01, 0.02, 0.03])

    def test_accepts_canonical_lowercase_device_headers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "lowercase.csv"
            _write_imu_csv(csv_path, device_prefix="device")

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=False,
            )

            self.assertTrue(set(required_imu_columns()).issubset(loaded.dataframe.columns))

    def test_accepts_device_underscore_headers_and_normalizes_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "device_underscore.csv"
            _write_imu_csv(csv_path, device_prefix="Device_")

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=False,
            )

            self.assertTrue(set(required_imu_columns()).issubset(loaded.dataframe.columns))
            self.assertFalse(any(column.startswith("Device") for column in loaded.dataframe.columns))

    def test_accepts_orbix_node_headers_and_normalizes_them(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "orbix_node.csv"
            _write_imu_csv(csv_path, device_prefix="OrbiX-Node-")

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=False,
            )

            self.assertTrue(set(required_imu_columns()).issubset(loaded.dataframe.columns))
            self.assertFalse(
                any(column.startswith("OrbiX-Node-") for column in loaded.dataframe.columns)
            )

    def test_accepts_mixed_device_header_casing(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "mixed.csv"
            columns = required_imu_columns()
            data = {
                (column.replace("device", "Device_", 1) if index % 2 else column): [float(index)]
                for index, column in enumerate(columns)
            }
            pd.DataFrame(data).to_csv(csv_path, index=False)

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=False,
            )

            self.assertTrue(set(columns).issubset(loaded.dataframe.columns))

    def test_partial_imu_mode_preserves_missing_columns_without_weakening_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "partial.csv"
            _write_imu_csv(csv_path, device_prefix="OrbiX-Node-")
            frame = pd.read_csv(csv_path)
            missing_prefix = "OrbiX-Node-5_IMU3_"
            frame = frame.drop(
                columns=[
                    column
                    for column in frame.columns
                    if column.startswith(missing_prefix)
                ]
            )
            frame.to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required IMU columns"):
                load_and_validate_input(
                    str(csv_path),
                    sample_rate_hz=100.0,
                    include_time_seconds=False,
                )

            loaded = load_and_validate_input(
                str(csv_path),
                sample_rate_hz=100.0,
                include_time_seconds=False,
                allow_partial_imu=True,
            )

            self.assertIn("device9_IMU3_GX", loaded.dataframe.columns)
            self.assertNotIn("device5_IMU3_GX", loaded.dataframe.columns)

    def test_rejects_duplicate_case_variants(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "ambiguous.csv"
            _write_imu_csv(csv_path, device_prefix="device")
            frame = pd.read_csv(csv_path)
            canonical = required_imu_columns()[0]
            frame[canonical.replace("device", "Device_", 1)] = frame[canonical]
            frame.to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "multiple header variants"):
                load_and_validate_input(
                    str(csv_path),
                    sample_rate_hz=100.0,
                    include_time_seconds=False,
                )


if __name__ == "__main__":
    unittest.main()
