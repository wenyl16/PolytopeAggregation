"""Regenerate the seven original-manuscript PDFs from current saved results.

Run from code: python -m Simulator.drawers.refresh_original_figures
Use --output-dir to render a review copy elsewhere. All figures are staged and
validated before replacing any destination. Existing target PDFs are archived
in the project tmp directory, and no simulation/training/evaluation is rerun.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from .plot_original_comparison import generate_comparison
from .plot_original_training import generate_training_figures
from .visualize_A_matrix_evolution import generate_heatmaps


# Preserve the user-visible Windows drive spelling for filesystem writes.
LEXICAL_CODE_ROOT = Path(__file__).absolute().parents[2]
WORKSPACE_ROOT = LEXICAL_CODE_ROOT.parent
EXPECTED_PDFS = {
    "aggregation_training_trajectory_small_scale_PEB.pdf",
    "aggregation_training_trajectory_small_scale_PB.pdf",
    "aggregation_training_trajectory_large_scale.pdf",
    "mg_training_trajectory.pdf",
    "A_difference_heatmap_cont_PEB.pdf",
    "A_difference_heatmap_cont_PB.pdf",
    "comparison.pdf",
}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path):
    path = Path(path).absolute()
    try:
        return path.relative_to(LEXICAL_CODE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _input_sources(records):
    """Normalize the three drawers' provenance formats to unique file hashes."""
    inputs = {}
    for record in records:
        for name in ("source_sha256", "plot_source_sha256"):
            for path, checksum in record.get(name, {}).items():
                if path in inputs and inputs[path] != checksum:
                    raise RuntimeError(f"An input changed between figure reads: {path}")
                inputs[path] = checksum
        sources = record.get("sources", [])
        sources = sources.values() if isinstance(sources, dict) else sources
        for source in sources:
            path, checksum = source["path"], source["sha256"]
            if path in inputs and inputs[path] != checksum:
                raise RuntimeError(f"An input changed between figure reads: {path}")
            inputs[path] = checksum
    for relative, checksum in inputs.items():
        path = Path(relative)
        path = path if path.is_absolute() else LEXICAL_CODE_ROOT / path
        if _digest(path) != checksum:
            raise RuntimeError(f"An input changed during figure generation: {path}")
    return inputs


def _publish(staged, output, records):
    existing = {name: (output / name).read_bytes() for name in EXPECTED_PDFS if (output / name).is_file()}
    archive_path = None
    if existing:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archive_path = WORKSPACE_ROOT / "tmp" / f"figures_original_before_refresh_{stamp}.zip"
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in existing.items():
                archive.writestr(name, content)
            previous = output / "refresh_manifest.json"
            if previous.is_file():
                archive.writestr(previous.name, previous.read_bytes())
    changed = []
    try:
        for name in sorted(EXPECTED_PDFS):
            content = (staged / name).read_bytes()
            changed.append(name)
            (output / name).write_bytes(content)
            if (output / name).read_bytes() != content:
                raise RuntimeError(f"Published PDF differs from its reviewed render: {name}")
    except Exception:
        for name in changed:
            if name in existing:
                (output / name).write_bytes(existing[name])
            else:
                (output / name).unlink(missing_ok=True)
        raise
    for record in records:
        name = Path(record.get("output_path", record.get("output"))).name
        record.pop("output", None)
        record["output_path"] = _display_path(output / name)
        record["pdf_sha256"] = _digest(output / name)
    return archive_path


def refresh_original_figures(output_dir=None):
    output = Path(output_dir or LEXICAL_CODE_ROOT / "results/figures_original").absolute()
    if output.is_symlink():
        raise ValueError("The output directory must not be a symbolic link")
    for name in EXPECTED_PDFS | {"refresh_manifest.json"}:
        if (output / name).is_symlink():
            raise ValueError(f"A figure destination cannot be a symbolic link: {name}")
    temp_root = WORKSPACE_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="original_figures_render_", dir=temp_root) as temporary:
        staged = Path(temporary)
        records = [*generate_training_figures(staged), *generate_heatmaps(staged),
                   generate_comparison(staged)]
        names = [Path(record.get("output_path", record.get("output"))).name for record in records]
        if len(names) != 7 or set(names) != EXPECTED_PDFS:
            raise RuntimeError("The restored figure set must contain exactly the seven original filenames")
        for name in names:
            pdf = staged / name
            if pdf.stat().st_size < 1000 or not pdf.read_bytes().startswith(b"%PDF-"):
                raise RuntimeError(f"Invalid generated PDF: {name}")
        input_sha256 = _input_sources(records)
        output.mkdir(parents=True, exist_ok=True)
        archive = _publish(staged, output, records)
    script_paths = [Path(__file__), Path(__file__).with_name("plot_original_training.py"),
                    Path(__file__).with_name("visualize_A_matrix_evolution.py"),
                    Path(__file__).with_name("plot_original_comparison.py"),
                    Path(__file__).with_name("plot_training.py")]
    legacy_paths = sorted({record["restored_from"] for record in records})
    manifest = {
        "status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_count": 7, "backend": "Python / Matplotlib", "formats": ["pdf"],
        "operation": "Regenerate original-manuscript figures from retained current result arrays",
        "training_or_solving_performed": False, "source_arrays_unchanged_verified": True,
        "sources_sha256": input_sha256,
        "plotting_sources_sha256": {_display_path(path): _digest(path) for path in script_paths},
        # Archived sources document the restoration but are not runtime
        # dependencies after the drawers have been recovered.
        "legacy_sources_sha256": {
            path: _digest(LEXICAL_CODE_ROOT / path) if (LEXICAL_CODE_ROOT / path).is_file() else None
            for path in legacy_paths
        },
        "previous_pdfs_backup": _display_path(archive) if archive else None,
        "notes": [
            "T24 formal small-scene exact PEB/PB runs supply both small trajectories and matrix differences.",
            "The seven-device main microgrid model is used; scale3 belongs to the revision experiments.",
            "The original large-scale trajectory uses T24 with 100 CL, 200 TCL and 200 PV; T96 is a separate revision experiment.",
            "comparison.pdf keeps the original five methods, using the latest completed common-direction evaluation.",
            "PEB has 94 rows; matrix ticks are based on actual saved dimensions.",
            "Heatmap color limits cover both matrix differences without clipping.",
            "Training display floor 1e-6 retains the original zero tick label; it does not certify zero raw error.",
        ],
        "figures": records,
    }
    manifest_path = output / "refresh_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    print(f"Updated all seven original figures: {output}")
    print(f"Saved input/output provenance: {manifest_path}")
    if archive:
        print(f"Previous PDFs archived: {archive}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        help="Override the default results/figures_original destination")
    args = parser.parse_args(argv)
    refresh_original_figures(args.output_dir)


if __name__ == "__main__":
    main()
