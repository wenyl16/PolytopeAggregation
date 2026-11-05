"""Restore the four original-manuscript training trajectories from retained runs.

The plotting logic was recovered from the v6 aggregation/microgrid training
drawers.  Their visual style is already implemented by ``plot_training``;
this entry point reconnects it to the current main results, validates saved
phase counts, and keeps the original manuscript filenames.  It performs no
training or optimization and does not import the archived v6 package.

Figure contract: show the two actual squared-distance training histories and
their trailing means, with the three saved training stages.  Preserve all
recorded updates and the v6 PDF style.  The 1e-6 display floor is not a zero-
error certificate; smoothing uses the original, unfloored observations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from Simulator.paths import CODE_ROOT, RESULTS_ROOT, comparison_output_path
from . import plot_training


TRAINING_FIGURES = (
    (
        "aggregation_training_trajectory_small_scale_PEB.pdf",
        "aggregation/T24/exact/peb",
        "Simulator_v6/drawers/plot_aggregation_training.py",
    ),
    (
        "aggregation_training_trajectory_small_scale_PB.pdf",
        "aggregation/T24/exact/pb",
        "Simulator_v6/drawers/plot_aggregation_training.py",
    ),
    (
        "aggregation_training_trajectory_large_scale.pdf",
        "aggregation/T24/large_scale/exact/peb",
        "Simulator_v6/drawers/plot_aggregation_training.py",
    ),
    (
        "mg_training_trajectory.pdf",
        "microgrid/training",
        "Simulator_v6/drawers/plot_mg_training.py",
    ),
)

# Resource count and time resolution are independent experiment dimensions.
# Validate the physical scene instead of relying on filenames such as "large".
EXPECTED_AGGREGATION_SCENARIOS = {
    "aggregation_training_trajectory_small_scale_PEB.pdf": (24, [5, 40, 5]),
    "aggregation_training_trajectory_small_scale_PB.pdf": (24, [5, 40, 5]),
    "aggregation_training_trajectory_large_scale.pdf": (24, [100, 200, 200]),
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_relative(path):
    return Path(path).resolve().relative_to(CODE_ROOT.resolve()).as_posix()


def _load_run(relative):
    directory = RESULTS_ROOT / relative
    history_path = directory / "error_history.pkl"
    config_path = directory / "training_config.json"
    if not config_path.is_file():
        config_path = directory / "migration_info.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    if configuration.get("status") not in ("completed", "retained_historical"):
        raise ValueError(f"Training is not complete: {config_path}")
    with history_path.open("rb") as stream:
        history = pickle.load(stream)
    values = [np.asarray(history[key], dtype=float).reshape(-1)
              for key in ("feasibility_errors", "optimality_errors")]
    if (not len(values[0]) or len(values[0]) != len(values[1])
            or not all(np.isfinite(value).all() and (value >= 0).all() for value in values)):
        raise ValueError(f"Invalid real-error history: {history_path}")
    updates = len(values[0])
    ends = configuration.get("phase_end_updates")
    if ends is None:
        ends = np.cumsum([call["optimizer_updates"]
                          for call in configuration.get("train_calls", [])]).tolist()
    if (len(ends) != 3 or any(type(end) is not int for end in ends)
            or ends != sorted(set(ends)) or ends[0] <= 0 or ends[-1] != updates):
        raise ValueError(f"Saved three-stage counts do not match the history: {config_path}")
    recorded = configuration.get("optimizer_updates", configuration.get("recorded_updates", updates))
    if recorded != updates:
        raise ValueError(f"Saved update total does not match the history: {config_path}")
    if list(plot_training.stage_boundaries(directory)) != ends[:-1]:
        raise ValueError(f"Plotting boundaries differ from the validated protocol: {directory}")
    return {
        "directory": directory,
        "history_path": history_path,
        "config_path": config_path,
        "config": configuration,
        "updates": updates,
        "phase_end_updates": ends,
        "source_sha256": {
            _code_relative(history_path): _sha256(history_path),
            _code_relative(config_path): _sha256(config_path),
        },
    }


def generate_training_figures(output_dir=None, *, window=20):
    """Draw four PDFs and return JSON-serializable source/plot metadata per PDF.

    Sources are the main T24 exact runs, the distinct 500-resource T24 run,
    and the current seven-device microgrid run. T96 and the scale3 sensitivity
    run are separate revision experiments. ``output_dir`` may select a staging
    directory under the project ``tmp`` tree for review before publication.
    """
    if not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    # Keep the caller's workspace drive spelling for writes.  On Windows a
    # mapped/junction workspace can resolve to another drive that is read-only
    # in the execution sandbox; the resolved path is used only for validation.
    lexical_code_root = Path(__file__).absolute().parents[2]
    output = Path(output_dir or lexical_code_root / "results" / "figures_original").absolute()
    comparison_output_path(output)
    # Validate all four inputs before producing the first output.
    runs = [_load_run(relative) for _name, relative, _legacy in TRAINING_FIGURES]
    for (filename, _relative, _legacy), run in zip(TRAINING_FIGURES, runs):
        if filename in EXPECTED_AGGREGATION_SCENARIOS:
            horizon, counts = EXPECTED_AGGREGATION_SCENARIOS[filename]
            config = run["config"]
            if config.get("horizon") != horizon or config.get("counts") != counts:
                raise ValueError(f"Wrong resource/time-resolution experiment for {filename}: "
                                 f"expected T={horizon}, CL/TCL/PV={counts}")
            scene_path = run["directory"] / "scenario.json"
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            actual_counts = [len(scene[key]) for key in ("CL_list", "TCL_list", "PV_list")]
            if (scene.get("T") != horizon or actual_counts != counts
                    or scene.get("scenario_sha256") != config.get("scenario_sha256")):
                raise ValueError(f"Saved physical devices disagree with the figure scenario: {filename}")
            final_path = run["directory"] / "final_polytope.npz"
            if _sha256(final_path) != config.get("final_polytope_sha256"):
                raise ValueError(f"Final model changed after training: {filename}")
            with np.load(final_path, allow_pickle=False) as final:
                if final["A"].shape[1] != horizon:
                    raise ValueError(f"Wrong matrix time dimension for {filename}")
            for path in (scene_path, final_path):
                run["source_sha256"][_code_relative(path)] = _sha256(path)
            origin_path = run["directory"] / "original_experiment_provenance.json"
            if origin_path.is_file():
                run["source_sha256"][_code_relative(origin_path)] = _sha256(origin_path)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for (filename, _relative, legacy), run in zip(TRAINING_FIGURES, runs):
        config = run["config"]
        counts = config.get("counts")
        scenario_label = (
            f"T = {config['horizon']}; {counts[1]} TCLs, {counts[2]} PVs and {counts[0]} CLs "
            f"({sum(counts)} resources)."
            if filename == "aggregation_training_trajectory_large_scale.pdf" else None
        )
        pdf = plot_training.plot_runs(
            [run["directory"]], output_dir=output, window=window,
            style="v6", output_name=Path(filename).stem,
            scenario_label=scenario_label,
        )
        records.append({
            "output_path": str(pdf.absolute()),
            "filename": filename,
            "source_directory": _code_relative(run["directory"]),
            "source_history": _code_relative(run["history_path"]),
            "source_config": _code_relative(run["config_path"]),
            "source_sha256": run["source_sha256"],
            "restored_from": legacy,
            "plotting_implementation": "Simulator.drawers.plot_training.plot_runs(style='v6')",
            "plot_source_sha256": {
                _code_relative(Path(__file__)): _sha256(Path(__file__)),
                _code_relative(Path(plot_training.__file__)): _sha256(Path(plot_training.__file__)),
            },
            "source_status": config["status"],
            "gradient_mode": config.get("gradient_mode"),
            "horizon": config.get("horizon"),
            "dtype": config.get("dtype"),
            "device_counts": config.get("device_counts", config.get("counts")),
            "scenario_sha256": config.get("scenario_sha256"),
            "conservativeness": config.get("conservativeness"),
            "optimizer_updates": run["updates"],
            "phase_end_updates": run["phase_end_updates"],
            "phase_updates": np.diff([0, *run["phase_end_updates"]]).tolist(),
            "stages": config.get("stages"),
            "moving_average_window": window,
            "scenario_label": scenario_label,
            "smoothing": "trailing arithmetic mean of original unfloored real errors",
            "display_floor": plot_training.DISPLAY_FLOOR,
            "display_floor_tick_label": "0",
            "all_recorded_updates_displayed": True,
            "metric": "sum of squared distances in normalized aggregate-power coordinates",
            "historical_protocol_limit": (
                "T96 retains histories and phase counts but lacks the original full training configuration."
                if config["status"] == "retained_historical" else None
            ),
            "pdf_sha256": _sha256(pdf),
        })
    for run in runs:
        for relative, checksum in run["source_sha256"].items():
            if _sha256(CODE_ROOT / relative) != checksum:
                raise RuntimeError(f"Training input changed while drawing: {relative}")
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args(argv)
    records = generate_training_figures(args.output_dir, window=args.window)
    print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
