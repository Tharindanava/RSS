from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rss.quaternion_series import normalize_quaternion_series, quaternion_to_axis_angle_series


class QuaternionSeriesTests(unittest.TestCase):
    def test_normalize_gives_unit_quaternions_with_nonnegative_scalar(self) -> None:
        rng = np.random.default_rng(123)
        q = rng.normal(size=(128, 4))
        q[17, :] = np.array([-1.0, 0.0, 0.0, 0.0])
        out = normalize_quaternion_series(q)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.all(out[:, 0] >= 0.0))
        np.testing.assert_allclose(out[17], [1.0, 0.0, 0.0, 0.0])

    def test_zero_quaternion_becomes_identity(self) -> None:
        out = normalize_quaternion_series(np.zeros((3, 4)))
        np.testing.assert_allclose(out, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))

    def test_axis_angle_recovers_known_rotations(self) -> None:
        rng = np.random.default_rng(456)
        axis = rng.normal(size=(64, 3))
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        angle = rng.uniform(0.05, 3.0, size=64)
        q = np.concatenate((np.cos(angle / 2)[:, None], axis * np.sin(angle / 2)[:, None]), axis=1)
        got_angle, got_axis = quaternion_to_axis_angle_series(q)
        np.testing.assert_allclose(got_angle, angle, atol=1e-10)
        np.testing.assert_allclose(got_axis, axis, atol=1e-10)

    def test_identity_rotation_has_zero_angle_and_a_defined_axis(self) -> None:
        angle, axis = quaternion_to_axis_angle_series(np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))
        np.testing.assert_allclose(angle, 0.0)
        np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1.0)


if __name__ == "__main__":
    unittest.main()
