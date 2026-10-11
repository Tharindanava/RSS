"""Dash entrypoint for the HMA Dashboard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dash import Dash

from HMA_Dashboard import catalog
from HMA_Dashboard import jobs
from HMA_Dashboard import raw_catalog
from HMA_Dashboard import signals
from HMA_Dashboard import te_matrix
from HMA_Dashboard import transfer_entropy
from HMA_Dashboard.callbacks import register_callbacks
from HMA_Dashboard.layout import build_layout


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_GAIT_ANALYSIS_RESULTS = PROJECT_ROOT / "outputs" / "gait_analysis"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
DEFAULT_DEBUG = False
DEFAULT_DEFAULT_FS: float | None = 100.0
DEFAULT_SAMPLE_DATA_DIR = PROJECT_ROOT / "Sample_Data"
DEFAULT_MAX_UPLOAD_BYTES = 32 * 1024 * 1024
#: Slightly above the client-side upload cap, so an oversize POST gets a clean
#: 413 instead of being buffered in full by a LAN-exposed dev server.
DEFAULT_MAX_CONTENT_LENGTH = 64 * 1024 * 1024


def create_app(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    default_fs: float | None = DEFAULT_DEFAULT_FS,
    *,
    sample_data_dir: str | Path = DEFAULT_SAMPLE_DATA_DIR,
    gait_analysis_results: str | Path = DEFAULT_GAIT_ANALYSIS_RESULTS,
    project_root: str | Path = PROJECT_ROOT,
    python_bin: str = jobs.DEFAULT_PYTHON_BIN,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_concurrent_jobs: int = jobs.MAX_CONCURRENT_JOBS,
    enable_pipeline_tab: bool = True,
) -> Dash:
    """Create and configure the HMA Dashboard app."""
    resolved_results = Path(results_dir).expanduser().resolve()
    resolved_sample_data = Path(sample_data_dir).expanduser().resolve()
    resolved_gait_analysis = Path(gait_analysis_results).expanduser().resolve()
    signals.configure_analysis_results_dir(resolved_gait_analysis)
    catalog_data = catalog.scan_results_catalog(results_dir)
    raw_catalog_data = raw_catalog.scan_raw_catalog(resolved_sample_data, resolved_results)
    app = Dash(
        __name__,
        title="HMA Dashboard",
        suppress_callback_exceptions=True,
        assets_folder=str(Path(__file__).resolve().parent / "assets"),
    )
    app.server.config["MAX_CONTENT_LENGTH"] = DEFAULT_MAX_CONTENT_LENGTH
    app.layout = build_layout(
        catalog_data=catalog_data,
        results_dir=resolved_results,
        default_fs=default_fs,
        raw_catalog_data=raw_catalog_data,
        run_settings={
            "results_dir": str(resolved_results),
            "default_fs": default_fs,
            "sample_data_dir": str(resolved_sample_data),
            "gait_analysis_results": str(resolved_gait_analysis),
            "project_root": str(Path(project_root).expanduser().resolve()),
            "python_bin": str(python_bin),
            "jobs_dir": str(jobs.jobs_dir_for(resolved_results)),
            "max_upload_bytes": int(max_upload_bytes),
            "max_concurrent_jobs": int(max_concurrent_jobs),
            "pipeline_enabled": bool(enable_pipeline_tab),
        },
    )
    register_callbacks(app)
    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HMA Dashboard.")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Root directory containing output runs.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Dash server host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Dash server port.")
    parser.add_argument("--debug", action="store_true", default=DEFAULT_DEBUG, help="Run the server in debug mode.")
    parser.add_argument(
        "--default-fs",
        type=float,
        default=DEFAULT_DEFAULT_FS,
        help="Optional sampling frequency used when time values are unavailable.",
    )
    parser.add_argument(
        "--sample-data-dir",
        type=str,
        default=str(DEFAULT_SAMPLE_DATA_DIR),
        help="Root directory containing raw IMU input CSV files.",
    )
    parser.add_argument(
        "--gait-analysis-results",
        type=str,
        default=str(DEFAULT_GAIT_ANALYSIS_RESULTS),
        help="Root containing gait_events.csv/gait_steps.csv analysis runs.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=jobs.DEFAULT_PYTHON_BIN,
        help="Python interpreter used to launch RSS pipeline jobs.",
    )
    parser.add_argument(
        "--max-upload-mb",
        type=int,
        default=DEFAULT_MAX_UPLOAD_BYTES // (1024 * 1024),
        help="Largest raw CSV accepted through the upload control.",
    )
    parser.add_argument(
        "--max-concurrent-jobs",
        type=int,
        default=jobs.MAX_CONCURRENT_JOBS,
        help="How many RSS pipeline jobs may run at once.",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Hide the Data & Runs tab, so the dashboard cannot write files or start jobs.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the dashboard, load one run from each available family, and exit.",
    )
    return parser.parse_args()


def run_smoke_test(
    results_dir: str | Path,
    default_fs: float | None,
    *,
    sample_data_dir: str | Path = DEFAULT_SAMPLE_DATA_DIR,
    gait_analysis_results: str | Path = DEFAULT_GAIT_ANALYSIS_RESULTS,
) -> None:
    """Run a lightweight dashboard smoke test."""
    catalog.clear_caches()
    signals.clear_caches()
    transfer_entropy.clear_caches()
    te_matrix.clear_caches()
    raw_catalog.clear_caches()

    catalog_data = catalog.scan_results_catalog(results_dir)
    app = create_app(
        results_dir=results_dir, default_fs=default_fs, sample_data_dir=sample_data_dir,
        gait_analysis_results=gait_analysis_results,
    )
    if app.layout is None:
        raise RuntimeError("Dash layout could not be built.")

    raw_data = raw_catalog.scan_raw_catalog(sample_data_dir, results_dir)
    if raw_data["exists"]:
        # Build (never run) one job command, so a broken preset or config schema
        # fails here rather than on someone's first click.
        unprocessed = raw_catalog.get_raw_file_options(raw_data, unprocessed_only=True)
        candidates = unprocessed or raw_catalog.get_raw_file_options(raw_data, unprocessed_only=False)
        if candidates:
            record = raw_catalog.get_raw_file_record(raw_data, candidates[0]["value"])
            request = jobs.build_job_request(
                csv_path=record["abs_path"],
                output_root=Path(results_dir) / "rss" / "smoke_test",
                results_dir=results_dir,
                project_root=PROJECT_ROOT,
                epochs=1,
            )
            argv = jobs.build_preset_argv(request, config_path="/dev/null")
            payload = jobs.build_config_payload(request)
            if "--quiet" in argv or len(payload) != 1 or set(payload[0]) != {"csv_path", "compute_device"}:
                raise RuntimeError("RSS job command or config payload is malformed.")
        print(
            f"Raw data: {len(raw_data['files'])} file(s), "
            f"{raw_data['processed_count']} processed, {raw_data['unprocessed_count']} unprocessed."
        )

    loaded: list[str] = []
    for family in ("joint", "limb"):
        run_id = catalog.get_first_run_id(catalog_data, family=family)
        if run_id is None:
            continue
        record = catalog.get_run_record(catalog_data, run_id)
        if record is None or not record["available_groups"]:
            continue
        group_name = str(record["available_groups"][0])
        signals.load_group_data(record["mode_file_path"], group_name, default_fs=default_fs)
        loaded.append(f"{family}:{group_name}")

    if not loaded:
        raise RuntimeError("Smoke test could not find any dashboard-compatible runs.")
    print("Smoke test passed:", ", ".join(loaded))


def _build_run_options(*, debug: bool, os_name: str | None = None) -> dict[str, Any]:
    """Build Dash run options with a Windows-safe debug configuration."""
    resolved_os_name = str(os_name or os.name)
    options: dict[str, Any] = {"debug": bool(debug)}
    if debug and resolved_os_name == "nt":
        # Werkzeug's reloader can trip WinError 10038 on Windows/Python 3.13.
        options["use_reloader"] = False
        options["dev_tools_hot_reload"] = True
    return options


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(
            results_dir=args.results_dir,
            default_fs=args.default_fs,
            sample_data_dir=args.sample_data_dir,
            gait_analysis_results=args.gait_analysis_results,
        )
        return

    app = create_app(
        results_dir=args.results_dir,
        default_fs=args.default_fs,
        sample_data_dir=args.sample_data_dir,
        gait_analysis_results=args.gait_analysis_results,
        python_bin=args.python_bin,
        max_upload_bytes=max(1, int(args.max_upload_mb)) * 1024 * 1024,
        max_concurrent_jobs=max(1, int(args.max_concurrent_jobs)),
        enable_pipeline_tab=not bool(args.read_only),
    )
    run_options = _build_run_options(debug=bool(args.debug))
    if bool(args.debug) and os.name == "nt":
        print("Debug mode enabled with Windows-safe reloader settings.")
    if args.read_only:
        print("Read-only mode: the Data & Runs tab is disabled.")
    else:
        print(f"Pipeline jobs: {jobs.jobs_dir_for(args.results_dir)}")
        if bool(args.debug) and os.name != "nt":
            print(
                "Debug mode uses the Werkzeug reloader; running jobs survive a restart "
                "and are recovered from that directory."
            )
    app.run(host=args.host, port=int(args.port), **run_options)


if __name__ == "__main__":
    main()
