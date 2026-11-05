"""Read and validate the retained paired gradient cohort without solver imports.

Each training seed is one paired replicate. Held-out directions are shared
within and across replicates; they must not be treated as independent seeds.
Only saved numerical data are read. No input or evaluation file is changed.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Simulator.paths import gradient_result_dir


SEEDS = tuple(range(0, 1000, 100))
METHODS = ("approximate", "exact")
UPDATES = 1000
EVALUATION_SAMPLES = 10000
EVALUATION_SEED = 10000
FEASIBILITY_THRESHOLD = 1e-4
STAGE_UPDATES = (300, 200, 500)
GRADIENT_CHECKPOINTS = (1, 100, 300, 500, 1000)
METRIC_NAMES = (
    "mean_feasibility", "max_feasibility", "feasibility_count_gt_1e_4",
    "mean_optimality", "max_optimality", "training_seconds",
)
METRIC_COLUMNS = (
    "training_seed", "evaluation_seed", "samples", "run", "evaluation",
    *(f"{method}_{metric}" for method in METHODS for metric in METRIC_NAMES),
)
GRADIENT_COLUMNS = (
    "seed", "iteration", "kind", "true_error", "exact_norm", "approximate_norm",
    "relative_gradient_difference", "cosine", "gradient_status",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cohort_directory(path):
    requested = Path(path) if path is not None else gradient_result_dir()
    requested = requested.resolve(strict=True)
    _require(requested.is_dir(), f"Expected a cohort directory: {requested}")
    if (requested / "batch_status.json").is_file():
        return requested
    candidates = [
        child for child in requested.iterdir()
        if child.is_dir() and (
            child.name.startswith("run_")
            or (child / "batch_status.json").is_file()
            or (child / "experiment_protocol.json").is_file()
            or (child / "summary" / "comparison.csv").is_file()
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one gradient cohort below {requested}; found "
            f"{len(candidates)}. Pass the intended run directory explicitly."
        )
    return candidates[0].resolve(strict=True)


class _Reader:
    def __init__(self, root):
        self.root = root
        self.hashes = {}

    def content(self, relative):
        path = self.root / relative
        content = path.read_bytes()
        self.hashes[path.relative_to(self.root).as_posix()] = hashlib.sha256(content).hexdigest()
        return content

    def json(self, relative):
        return json.loads(self.content(relative).decode("utf-8-sig"))

    def csv(self, relative):
        return pd.read_csv(io.BytesIO(self.content(relative)), encoding="utf-8-sig")

    def array(self, relative):
        values = np.load(io.BytesIO(self.content(relative)), allow_pickle=False)
        _require(isinstance(values, np.ndarray), f"Expected NPY array: {relative}")
        return values

    def archive(self, relative):
        with np.load(io.BytesIO(self.content(relative)), allow_pickle=False) as archive:
            return {name: archive[name].copy() for name in archive.files}


def _numbers(values, shape, description):
    values = np.asarray(values, dtype=float)
    _require(values.shape == shape and np.all(np.isfinite(values)),
             f"Unexpected shape or nonfinite values in {description}; expected {shape}")
    return values


def _same(actual, expected, description):
    _require(np.array_equal(actual, expected), f"Mismatched {description}")


def _close(actual, expected, description):
    _require(np.isfinite(actual) and np.isclose(actual, expected, rtol=1e-11, atol=1e-14),
             f"Saved metric differs from raw data: {description}")


def _relative_directory(base, relative):
    # Saved Windows relative paths remain readable on other platforms.
    return (base / str(relative).replace("\\", "/")).resolve(strict=True)


def _gradient_rows(reader, relative, seed):
    saved = reader.json(relative)
    _require(isinstance(saved, list), f"Expected gradient records: {relative}")
    rows = []
    observed = set()
    expected = {(iteration, kind) for iteration in GRADIENT_CHECKPOINTS
                for kind in ("feasibility", "optimality")}
    for item in saved:
        row = {"seed": seed, **{key: item[key] for key in GRADIENT_COLUMNS if key != "seed"}}
        identity = (row["iteration"], row["kind"])
        _require(identity in expected and identity not in observed,
                 f"Unexpected or repeated gradient checkpoint: seed {seed}, {identity}")
        observed.add(identity)
        for key in ("true_error", "exact_norm", "approximate_norm"):
            value = row[key]
            _require(value is not None and np.isfinite(value) and value >= 0,
                     f"Invalid gradient {key}: seed {seed}, {identity}")
        difference = row["relative_gradient_difference"]
        _require(difference is None or (np.isfinite(difference) and difference >= 0),
                 f"Invalid relative gradient difference: seed {seed}, {identity}")
        cosine = row["cosine"]
        _require(cosine is None or (np.isfinite(cosine) and -1 - 1e-12 <= cosine <= 1 + 1e-12),
                 f"Invalid gradient cosine: seed {seed}, {identity}")
        _require(isinstance(row["gradient_status"], str) and row["gradient_status"],
                 f"Missing gradient status: seed {seed}, {identity}")
        rows.append(row)
    _require(observed == expected, f"Incomplete gradient checkpoints for seed {seed}")
    return rows


def load_cohort(path=None):
    """Return validated data for the ten retained 0,100,...,900 seed pairs.

    ``path`` may be the parent comparison directory or an explicit batch.
    A parent containing several batches is rejected rather than selecting one
    by date, name, completion status, or the numerical results. ``metrics``
    preserves all 17 columns of the saved comparison CSV. Undefined gradient
    ratios/cosines remain NaN in ``gradients``; no numerical floor is applied.
    """
    cohort = _cohort_directory(path)
    reader = _Reader(cohort)
    batch = reader.json("batch_status.json")
    _require(batch.get("status") == "completed", "The paired gradient batch is not complete")
    _same(batch.get("training_seeds"), SEEDS, "fixed batch training seeds")
    _require(batch.get("case_seed") == 0 and batch.get("actual_updates") == UPDATES
             and batch.get("evaluation_samples") == EVALUATION_SAMPLES
             and batch.get("evaluation_seed") == EVALUATION_SEED
             and batch.get("train_only") is False, "Unexpected paired batch protocol")
    runs = batch.get("runs", [])
    _require(len(runs) == len(SEEDS), "The batch must contain all ten planned seeds")
    for seed, record in zip(SEEDS, runs):
        _require(record.get("training_seed") == seed and record.get("status") == "completed"
                 and record.get("path") == f"seed_{seed}", f"Incomplete batch record for seed {seed}")

    protocol = reader.json("experiment_protocol.json")
    _same(protocol.get("training_seeds"), SEEDS, "predeclared training seeds")
    _same(protocol.get("stage_updates"), STAGE_UPDATES, "stage update counts")
    _require(protocol.get("evaluation_directions") == EVALUATION_SAMPLES
             and protocol.get("evaluation_seed") == EVALUATION_SEED,
             "Unexpected predeclared evaluation protocol")
    summary = reader.json("summary/summary.json")
    _require(summary.get("included_count") == len(SEEDS) and summary.get("excluded_count") == 0,
             "The summary must retain all ten paired seeds")
    metrics = reader.csv("summary/comparison.csv")
    _require(tuple(metrics.columns) == METRIC_COLUMNS,
             "Unexpected columns in summary/comparison.csv")
    _require(len(metrics) == len(SEEDS) and not metrics.training_seed.duplicated().any(),
             "Repeated or incomplete seed rows in summary/comparison.csv")
    metrics = metrics.sort_values("training_seed").reset_index(drop=True)
    _same(metrics.training_seed.to_numpy(), SEEDS, "summary training seeds")
    _same(metrics.evaluation_seed.to_numpy(), [EVALUATION_SEED] * len(SEEDS), "summary evaluation seeds")
    _same(metrics.samples.to_numpy(), [EVALUATION_SAMPLES] * len(SEEDS), "summary evaluation sizes")

    training = {method: {key: [] for key in
                ("feasibility_error", "optimality_error", "step_seconds")} for method in METHODS}
    gradients = []
    common_evaluation = np.random.RandomState(EVALUATION_SEED).normal(size=(EVALUATION_SAMPLES, 24))
    initial_reference = None
    source_reference = None
    expected_stages = np.repeat((1, 2, 3), STAGE_UPDATES)
    iterations = np.arange(1, UPDATES + 1)

    for seed, metric in zip(SEEDS, metrics.to_dict("records")):
        folder = f"seed_{seed}"
        directory = cohort / folder
        evaluation_folder = f"{folder}/evaluation_{EVALUATION_SAMPLES}_seed{EVALUATION_SEED}"
        _require(_relative_directory(cohort / "summary", metric["run"]) == directory,
                 f"Unexpected summary run path for seed {seed}")
        _require(_relative_directory(cohort / "summary", metric["evaluation"]) == cohort / evaluation_folder,
                 f"Unexpected summary evaluation path for seed {seed}")
        meta = reader.json(f"{folder}/metadata.json")
        _require(meta.get("status") == "completed" and meta.get("training_seed") == seed
                 and meta.get("case_seed") == 0 and meta.get("initialization_seed") == 0
                 and meta.get("iterations") == UPDATES
                 and meta.get("evaluation_seed") == EVALUATION_SEED
                 and meta.get("directions_identical_verified") is True,
                 f"Unexpected or incomplete training metadata for seed {seed}")
        _same([stage[0] for stage in meta.get("stages", [])], STAGE_UPDATES,
              f"training stage updates for seed {seed}")
        _same(meta.get("stages"), [[300, .02, 1.], [200, .1, .01], [500, 1., 1e-6]],
              f"lr=1 training schedule for seed {seed}")
        if source_reference is None:
            source_reference = meta.get("source_sha256")
        _require(source_reference and meta.get("source_sha256") == source_reference,
                 f"The paired seeds were generated with different training sources: seed {seed}")
        planned = _numbers(reader.array(f"{folder}/training_directions.npy"), (UPDATES, 24),
                           f"planned training directions for seed {seed}")
        _same(planned, np.random.RandomState(seed).normal(size=(UPDATES, 24)),
              f"declared training random stream for seed {seed}")
        _require(hashlib.sha256(planned.tobytes()).hexdigest() == meta["training_directions_sha256"],
                 f"Training direction hash differs for seed {seed}")
        for method in METHODS:
            used = reader.array(f"{folder}/{method}_used_directions.npy")
            _same(used, planned, f"{method} training directions for seed {seed}")
        initial = reader.archive(f"{folder}/initial_polytope.npz")
        for key, shape in (("A", (94, 24)), ("b", (94,))):
            _numbers(initial[key], shape, f"initial {key} for seed {seed}")
        if initial_reference is None:
            initial_reference = initial
        for key in ("A", "b"):
            _same(initial[key], initial_reference[key], f"common initial {key} for seed {seed}")

        history = reader.csv(f"{folder}/training.csv")
        _require(len(history) == 2 * UPDATES and set(history.method) == set(METHODS),
                 f"Incomplete paired training CSV for seed {seed}")
        for method in METHODS:
            method_rows = history.loc[history.method == method].sort_values("iteration")
            _same(method_rows.iteration.to_numpy(), iterations, f"{method} iterations for seed {seed}")
            _same(method_rows.stage.to_numpy(), expected_stages, f"{method} stages for seed {seed}")
            for key in training[method]:
                values = _numbers(method_rows[key].to_numpy(), (UPDATES,),
                                  f"{method} training {key} for seed {seed}")
                _require(np.all(values >= 0), f"Negative {key} in {method}, seed {seed}")
                training[method][key].append(values)

        evaluation = reader.json(f"{evaluation_folder}/metadata.json")
        _require(evaluation.get("status") == "completed"
                 and evaluation.get("samples") == EVALUATION_SAMPLES
                 and evaluation.get("direction_seed") == EVALUATION_SEED
                 and evaluation.get("case_seed") == 0
                 and evaluation.get("same_directions_both_methods") is True
                 and evaluation.get("final_polytopes_unchanged") is True
                 and evaluation.get("feasibility_count_threshold") == FEASIBILITY_THRESHOLD,
                 f"Unexpected or incomplete held-out evaluation for seed {seed}")
        directions = _numbers(reader.array(f"{evaluation_folder}/directions.npy"),
                              (EVALUATION_SAMPLES, 24), f"held-out directions for seed {seed}")
        _same(directions, common_evaluation, f"shared held-out directions for seed {seed}")
        _require(hashlib.sha256(directions.tobytes()).hexdigest() == evaluation["directions_sha256"],
                 f"Held-out direction hash differs for seed {seed}")
        for method in METHODS:
            reader.content(f"{folder}/{method}_final.npz")
            _require(reader.hashes[f"{folder}/{method}_final.npz"] ==
                     evaluation["final_polytope_sha256"][method],
                     f"Final {method} polytope differs from evaluated input for seed {seed}")
        raw = reader.csv(f"{evaluation_folder}/errors.csv")
        _require(len(raw) == EVALUATION_SAMPLES, f"Incomplete held-out errors for seed {seed}")
        _same(raw.direction_index.to_numpy(), np.arange(1, EVALUATION_SAMPLES + 1),
              f"held-out direction indices for seed {seed}")
        for method in METHODS:
            arrays = {kind: _numbers(raw[f"{method}_{kind}_error"].to_numpy(),
                                    (EVALUATION_SAMPLES,), f"{method} held-out {kind}, seed {seed}")
                      for kind in ("feasibility", "optimality")}
            _require(all(np.all(values >= 0) for values in arrays.values()),
                     f"Negative held-out errors for {method}, seed {seed}")
            recomputed = {
                "mean_feasibility": arrays["feasibility"].mean(),
                "max_feasibility": arrays["feasibility"].max(),
                "feasibility_count_gt_1e_4": np.count_nonzero(arrays["feasibility"] > FEASIBILITY_THRESHOLD),
                "mean_optimality": arrays["optimality"].mean(),
                "max_optimality": arrays["optimality"].max(),
                "training_seconds": training[method]["step_seconds"][-1].sum(),
            }
            for key, expected in recomputed.items():
                _close(metric[f"{method}_{key}"], expected, f"seed {seed}, {method} {key}")
        gradients.extend(_gradient_rows(reader, f"{folder}/same_point_gradients.json", seed))

    gradient_frame = pd.DataFrame(gradients, columns=GRADIENT_COLUMNS)
    for key in ("true_error", "exact_norm", "approximate_norm", "relative_gradient_difference", "cosine"):
        gradient_frame[key] = pd.to_numeric(gradient_frame[key], errors="raise")
    return {
        "cohort_dir": cohort,
        "seeds": list(SEEDS),
        "metrics": metrics,
        "training": {method: {key: np.stack(values) for key, values in series.items()}
                     for method, series in training.items()},
        "iterations": iterations,
        "stages": [300, 500],
        "eval_samples": EVALUATION_SAMPLES,
        "directions_seed": EVALUATION_SEED,
        "source_hashes": dict(sorted(reader.hashes.items())),
        "gradients": gradient_frame,
    }
