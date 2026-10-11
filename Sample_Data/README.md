# Sample_Data

Raw input recordings. Only the QArm practical-validation set is distributed:

```text
Sample_Data/QArm/
    session_2026-09-09/   7 batches (35 tests), wrist IMU = OrbiX-Node-4
    session_2026-09-18/   7 batches (batch 7 has no Node-10 gyro columns), wrist IMU = OrbiX-Node-10

each session folder contains
    setup.txt                                  which IMU node was mounted on the wrist / forearm / shoulder
    <batch>/combined_log_wide_<epoch>.csv      raw multi-IMU log, 100 Hz, five tests per batch
    Ground_Truth/results/test_N.csv            QArm encoder log (commanded and measured joint angles)
    Ground_Truth/results/test_N.json           parameters of the commanded multisine trajectory
```

See [`qarm_rss_experiments/README.md`](../qarm_rss_experiments/README.md) for the protocol and how to analyse it.
Column names follow `OrbiX-Node-<n>_IMU1_{AX,AY,AZ,GX,GY,GZ}` (accelerometer in the A columns, gyroscope in the G columns;
the QArm analysis uses the gyroscope of the wrist node in **deg/s**).

## Your own recordings

`HMA_Dashboard`, `hilbert_analysis` and `gait_analysis` expect human-motion recordings under
`Sample_Data/Human_Motion/<subject>/<terrain>/<experiment>/<recording>.csv` (wide-format logs with `device<k>_IMU<j>_{AX..GZ}`
columns; the segment ↔ sensor mapping is `SEGMENT_SENSOR_MAPPING` in `rss/sensor_layout.py`). `rss run-config` mirrors that
hierarchy under the output root, so RSS results end up in `outputs/rss/Human_Motion/<subject>/<terrain>/…`. The folder name
`Human_Motion` is the constant `HUMAN_MOTION_DIR` in `HMA_Dashboard/catalog.py` and `hilbert_analysis/pipeline.py`. No human
recordings are distributed with this repository; `scripts/make_demo_outputs.py` writes synthetic RSS-style *outputs* so the
analysis tools and the dashboard can be tried without them.
