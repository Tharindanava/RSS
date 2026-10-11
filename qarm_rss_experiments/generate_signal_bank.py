"""Command-line entry point for regenerating QArm RSS signal artifacts."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qarm_rss_experiments.signal_bank import save_signal_bank


def main() -> None:
    bank_path, manifest_path = save_signal_bank()
    print(f"Wrote signal bank: {bank_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
