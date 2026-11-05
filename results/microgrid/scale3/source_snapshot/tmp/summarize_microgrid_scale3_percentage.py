"""Derive the Figure 06 percentage metric from existing scale3 trajectories."""
from pathlib import Path
import hashlib
import json
import pickle

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "code/results/microgrid/scale3"
EVAL = BASE / "disaggregation"
sources = [EVAL / "results.pkl", EVAL / "metadata.json", EVAL / "errors.csv",
           BASE / "training/pretrainnet_weights.pth", BASE / "training/training_config.json"]
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
metadata = json.loads((EVAL / "metadata.json").read_text(encoding="utf-8"))
with (EVAL / "results.pkl").open("rb") as stream:
    rows = pickle.load(stream)
assert metadata["status"] == "completed" and len(rows) == 5000
assert all(row["solver_status"] == "completed" for row in rows)
command = np.asarray([row["power_curve"] for row in rows], dtype=float)
projected = np.asarray([row["projected_power_curve"] for row in rows], dtype=float)
physical_sse = np.asarray([row["feasibility_error"] for row in rows], dtype=float)
assert command.shape == projected.shape == (5000, 24)
assert np.isfinite(command).all() and np.isfinite(projected).all()
trajectory_sse = np.sum((command - projected) ** 2, axis=1)
np.testing.assert_allclose(physical_sse, trajectory_sse, rtol=1e-10, atol=1e-12)
denominator = np.sum(command ** 2, axis=1)
assert np.isfinite(physical_sse).all() and np.all(physical_sse >= 0)
assert np.isfinite(denominator).all() and np.all(denominator > 0)
percentages = 100 * np.sqrt(physical_sse / denominator)
summary = {
    "status": "completed", "method": "postprocess_existing_trajectories",
    "additional_solver_calls": 0, "T": 24, "device_counts": metadata["device_counts"],
    "training_conservativeness": metadata["training_conservativeness"],
    "evaluation_conservativeness": metadata["evaluation_conservativeness"],
    "seed": metadata["seed"], "noise_scale": metadata["noise_scale"],
    "mean_shifts": {name: metadata[name] for name in ("pv_shift", "cl_shift", "temp_shift")},
    "metric": "relative_disaggregation_error_percent", "unit": "percent",
    "formula": "100*sqrt(sum((P_command-P_projected)**2)/sum(P_command**2))",
    "projection_definition": "Reuse the original projection minimizing squared distance in fixed nominal normalized power coordinates.",
    "valid_samples": int(len(percentages)), "failed_samples": metadata["failed_samples"],
    "zero_denominator_samples": int(np.count_nonzero(denominator == 0)),
    "mean": float(percentages.mean()), "max": float(percentages.max()),
    "max_sample_index": int(rows[int(np.argmax(percentages))]["sample_index"]),
    "physical_squared_error_recompute_max_abs_difference": float(np.max(np.abs(physical_sse - trajectory_sse))),
    "source_sha256": hashes,
}
assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
(EVAL / "relative_disaggregation_summary.json").write_text(
    json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
print(json.dumps({key: summary[key] for key in ("valid_samples", "mean", "max", "additional_solver_calls")}, indent=2))
