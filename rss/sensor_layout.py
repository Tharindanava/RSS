"""Body-segment to IMU-channel layout of the wearable platform.

Each of the 18 body segments is observed by one IMU of the multi-node platform. A wide-format recording
holds six channels per sensor, named ``<device>_<IMU>_{AX,AY,AZ,GX,GY,GZ}`` (accelerometer, gyroscope).
Edit ``SEGMENT_SENSOR_MAPPING`` to match a different sensor placement.
"""

from __future__ import annotations

# Body segment -> ``<device>_<IMU>`` prefix of its channels in the recording.
SEGMENT_SENSOR_MAPPING: dict[str, str] = {
    "pelvis_to_lower_spine": "device5_IMU3",
    "mid_to_upper_spine": "device9_IMU3",
    "neck_to_right_shoulder": "device9_IMU2",
    "neck_to_left_shoulder": "device9_IMU1",
    "right_shoulder_to_elbow": "device4_IMU2",
    "right_elbow_to_wrist": "device7_IMU2",
    "right_wrist_to_hand": "device7_IMU1",
    "left_shoulder_to_elbow": "device2_IMU2",
    "left_elbow_to_wrist": "device6_IMU2",
    "left_wrist_to_hand": "device6_IMU1",
    "pelvis_to_right_hip": "device5_IMU2",
    "right_hip_to_knee": "device3_IMU2",
    "right_knee_to_ankle": "device8_IMU3",
    "right_ankle_to_foot": "device8_IMU1",
    "pelvis_to_left_hip": "device5_IMU1",
    "left_hip_to_knee": "device1_IMU2",
    "left_knee_to_ankle": "device10_IMU3",
    "left_ankle_to_foot": "device10_IMU1",
}


IMU_CHANNEL_SUFFIXES: tuple[str, ...] = ("AX", "AY", "AZ", "GX", "GY", "GZ")


def required_imu_columns() -> list[str]:
    """All IMU columns a full-body recording must contain, sorted."""
    return sorted({f"{sensor}_{suffix}" for sensor in SEGMENT_SENSOR_MAPPING.values() for suffix in IMU_CHANNEL_SUFFIXES})
