"""Paired bootstrap means for the saved gradient-comparison test directions.

Clouds describe sampling variation of each test-set mean, conditional on its
trained model. They are not the distribution of individual directional errors
or independent training repetitions. No original input is modified.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from Simulator.paths import RESULTS_ROOT
from ._gradient_comparison_data import METHODS


_VERSION = "paired-multinomial-direction-bootstrap-v1"
_BATCH_SIZE = 64


def _positive_integer(value, description, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}")
    return int(value)


def _raw_inputs(cohort):
    root = Path(cohort["cohort_dir"])
    seeds = cohort["seeds"]
    sample_count = int(cohort["eval_samples"])
    evaluation_seed = int(cohort["directions_seed"])
    matrices, stats, used_hashes = [], [], {}
    metrics = cohort["metrics"].set_index("training_seed")
    for training_seed in seeds:
        relative = (f"seed_{training_seed}/evaluation_{sample_count}_seed{evaluation_seed}"
                    "/errors.csv")
        content = (root / relative).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != cohort["source_hashes"].get(relative):
            raise ValueError(f"Raw test errors differ from validated inputs: {relative}")
        used_hashes[relative] = digest
        frame = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        if (len(frame) != sample_count or not np.array_equal(
                frame.direction_index.to_numpy(), np.arange(1, sample_count + 1))):
            raise ValueError(f"Raw direction indices are incomplete or reordered: {relative}")
        for method in METHODS:
            array = frame[[f"{method}_feasibility_error", f"{method}_optimality_error"]].to_numpy(dtype=float)
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError(f"Invalid directional errors: seed {training_seed}, {method}")
            center = np.array([metrics.loc[training_seed, f"{method}_mean_{kind}"]
                               for kind in ("feasibility", "optimality")], dtype=float)
            if not np.allclose(array.mean(axis=0), center, rtol=1e-11, atol=1e-14):
                raise ValueError(f"Raw means disagree with saved summary: seed {training_seed}, {method}")
            if center[0] <= 0:
                raise ValueError("Logarithmic mean-feasibility coordinates must be positive")
            matrices.append(array)
            record = {"seed": int(training_seed), "method": method, "n": sample_count}
            for column, kind in enumerate(("feasibility", "optimality")):
                values = array[:, column]
                record[kind] = {
                    "mean": float(values.mean()), "median": float(np.median(values)),
                    "minimum": float(values.min()), "maximum": float(values.max()),
                    "p10": float(np.quantile(values, .1)), "p80": float(np.quantile(values, .8)),
                    "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)),
                    "p99": float(np.quantile(values, .99)), "p999": float(np.quantile(values, .999)),
                }
            record["feasibility_count_strictly_above_1e4"] = int(np.count_nonzero(array[:, 0] > 1e-4))
            record["feasibility_fraction_at_or_below_1e12"] = float(np.mean(array[:, 0] <= 1e-12))
            stats.append(record)
    # Consecutive columns are [seed, method, coordinate], with directions in rows.
    return np.concatenate(matrices, axis=1), stats, used_hashes


def _bootstrap_means(raw, replicates, random_seed, seed_count):
    sample_count = len(raw)
    probabilities = np.full(sample_count, 1.0 / sample_count)
    rng = np.random.default_rng(random_seed)
    means = np.empty((replicates, raw.shape[1]), dtype=float)
    for start in range(0, replicates, _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, replicates)
        # One shared direction-count vector resamples every trained model.
        # Multinomial(n; 1/n,...,1/n) is the standard n-out-of-n bootstrap.
        counts = rng.multinomial(sample_count, probabilities, size=stop - start)
        if not np.all(counts.sum(axis=1) == sample_count):
            raise RuntimeError("Bootstrap counts do not contain exactly n directions")
        means[start:stop] = counts @ raw / sample_count
    return means.reshape(replicates, seed_count, len(METHODS), 2)


def _cache_path(cache_dir, signature):
    if cache_dir is None:
        return None
    directory = Path(cache_dir).resolve()
    if directory.is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError("Bootstrap cache must stay outside the formal results directory")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"bootstrap_means_{signature[:20]}.npz"


def _save_cache(path, means, signature, payload):
    handle = tempfile.NamedTemporaryFile(prefix="bootstrap_", suffix=".npz",
                                         dir=path.parent, delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            np.savez_compressed(handle, bootstrap_means=means, signature=np.array(signature),
                                signature_payload=np.array(json.dumps(payload, sort_keys=True)))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_bootstrap_clouds(cohort, replicates=1200, seed=20261025,
                           central_fraction=0.8, points_per_cloud=60, cache_dir=None):
    """Return real-coordinate paired bootstrap mean clouds for all seed pairs.

    The result's ``bootstrap_means`` has axes [replicate, training seed, method,
    coordinate], with method order ``METHODS`` and coordinate order [feasibility,
    optimality]. Every replicate uses the SAME multinomial direction counts for
    every seed and method, preserving the original common-test-direction pairing.

    Centrality is defined in [log10(mean feasibility), mean optimality] space.
    Its origin is the mean of the FULL bootstrap coordinate cloud; covariance
    also uses the full cloud. Exactly ceil(B*central_fraction) nearest points by
    Mahalanobis distance are retained, with stable replicate-order tie breaking.
    Display IDs are sampled uniformly without replacement from that central set
    using independent, deterministic RNG streams. Original seed means remain
    unchanged in ``center`` and are never used to recenter bootstrap samples.

    Caching stores only the bootstrap means. Its signature includes algorithm,
    B, RNG seed, seed/method order, sample count and validated input hashes.
    Presentation settings are recomputed from cached means for visual iteration.
    """
    replicates = _positive_integer(replicates, "replicates", minimum=20)
    random_seed = _positive_integer(seed, "seed", minimum=0)
    points_per_cloud = _positive_integer(points_per_cloud, "points_per_cloud")
    if (not np.isfinite(central_fraction) or not 0 < central_fraction <= 1):
        raise ValueError("central_fraction must satisfy 0 < value <= 1")
    central_fraction = float(central_fraction)
    started = time.perf_counter()
    raw, raw_stats, raw_hashes = _raw_inputs(cohort)
    seeds = list(cohort["seeds"])
    payload = {
        "algorithm": _VERSION, "replicates": replicates, "seed": random_seed,
        "training_seeds": [int(value) for value in seeds], "methods": list(METHODS),
        "sample_count": int(cohort["eval_samples"]), "batch_size": _BATCH_SIZE,
        "input_hashes": dict(sorted(cohort["source_hashes"].items())),
        "raw_error_hashes": raw_hashes,
    }
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = _cache_path(cache_dir, signature)
    cache_hit = path is not None and path.is_file()
    if cache_hit:
        with np.load(path, allow_pickle=False) as cached:
            if cached["signature"].item() != signature or json.loads(cached["signature_payload"].item()) != payload:
                raise ValueError(f"Bootstrap cache signature mismatch: {path}")
            means = cached["bootstrap_means"].copy()
    else:
        means = _bootstrap_means(raw, replicates, random_seed, len(seeds))
        if path is not None:
            _save_cache(path, means, signature, payload)
    expected_shape = (replicates, len(seeds), len(METHODS), 2)
    if means.shape != expected_shape or not np.isfinite(means).all() or np.any(means[..., 0] <= 0):
        raise ValueError("Bootstrap means have invalid shape or nonpositive logarithmic coordinates")
    metrics = cohort["metrics"].set_index("training_seed")
    clouds = []
    central_count = int(math.ceil(replicates * central_fraction))
    for seed_index, training_seed in enumerate(seeds):
        for method_index, method in enumerate(METHODS):
            sample_means = means[:, seed_index, method_index, :]
            log_coordinates = np.c_[np.log10(sample_means[:, 0]), sample_means[:, 1]]
            selection_center = log_coordinates.mean(axis=0)
            covariance = np.cov(log_coordinates, rowvar=False, ddof=1)
            residual = log_coordinates - selection_center
            inverse = np.linalg.pinv(covariance, rcond=1e-12, hermitian=True)
            squared = np.maximum(np.einsum("ni,ij,nj->n", residual, inverse, residual), 0)
            nearest = np.argsort(squared, kind="stable")[:central_count]
            central_ids = np.sort(nearest)
            selection_rng = np.random.default_rng(np.random.SeedSequence(
                [random_seed, 811, seed_index, method_index]))
            selected_ids = np.sort(selection_rng.choice(
                central_ids, size=min(points_per_cloud, central_count), replace=False))
            center = np.array([metrics.loc[training_seed, f"{method}_mean_{kind}"]
                               for kind in ("feasibility", "optimality")], dtype=float)
            clouds.append({
                "seed": int(training_seed), "method": method, "center": center,
                "means": sample_means, "central_ids": central_ids, "selected_ids": selected_ids,
                "coordinates_log": log_coordinates, "central_fraction": central_fraction,
                "central_fraction_effective": central_count / replicates,
                "selection_center_log": selection_center, "covariance_log": covariance,
                "cutoff_squared": float(squared[nearest[-1]]), "mahalanobis_squared": squared,
            })
    return {
        "bootstrap_means": means, "clouds": clouds, "raw_stats": raw_stats,
        "signature": signature, "signature_payload": payload,
        "cache_hit": cache_hit, "cache_path": str(path) if path is not None else None,
        "seconds": time.perf_counter() - started,
        "selection_definition": "nearest empirical Mahalanobis distances from the full bootstrap-cloud mean in log10(mean feasibility), mean optimality coordinates",
        "selected_point_definition": "uniform sample without replacement from each central set, using separate deterministic RNG streams",
    }
