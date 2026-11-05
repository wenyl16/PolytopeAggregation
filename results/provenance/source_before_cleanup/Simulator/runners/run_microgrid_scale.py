"""Finish one retained scaled Microgrid training/evaluation experiment.

Use --wait-training-pid to adopt an already-running training process. It is
never retrained by that path. Every stage uses the existing experiment entry
points and fixed result paths; no timestamped experiment directories are made.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


CODE = Path(__file__).resolve().parents[2]
ROOT = CODE.parent
VERIFIER = ROOT / "tmp/verify_microgrid_scale.py"
METRICS = ("feasibility_error", "feasibility_error_rated", "optimality_error_rated",
           "objective_error", "objective_error_rated")
FINAL_TRAINING_FILES = ("metadata.pkl", "training_config.json", "setup_timing.json", "error_history.pkl",
                        "pretrainnet_weights.pth", "final_polytope.npz", "solver_diagnostics.json")


def device_counts(device_scale):
    if not isinstance(device_scale, int) or isinstance(device_scale, bool) or device_scale not in (3, 5):
        raise ValueError("This pipeline supports device_scale 3 or 5")
    return {name: device_scale * (2 if name == "HP" else 1)
            for name in ("MT", "PV", "TL", "CL", "ESS", "HP")}


def scale_output(device_scale):
    device_counts(device_scale)
    return CODE / "results" / "microgrid" / f"scale{device_scale}"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    """Training/evaluation writers update small JSON files without rename."""
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            if attempt == 2:
                return None
            time.sleep(0.05)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # OneDrive can briefly hold an existing destination open on Windows.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_review(result_dir):
    """Attach optional diagnostics only when their original inputs still match."""
    certificate_path = result_dir / "attainability_recheck.json"
    projection_path = result_dir / "projection_recheck.json"
    witness_path = result_dir / "attainability_recheck.npz"
    certificate, projection = read_json(certificate_path), read_json(projection_path)
    if not certificate or not projection or not witness_path.is_file():
        return None
    if (certificate.get("status") != "completed_diagnostic"
            or not certificate.get("input_files_unchanged")
            or projection.get("status") != "completed" or not projection.get("inputs_unchanged")):
        return None
    hashes = {**certificate["input_sha256_before"], **projection["input_sha256"]}
    if not all((ROOT / name).is_file() and sha256(ROOT / name) == expected for name, expected in hashes.items()):
        return None
    if not (certificate.get("nominal_saved_y_all_certified")
            and certificate.get("perturbed_saved_x_all_certified")):
        return None
    if not all(any(attempt.get("target_kind") == "exact_fixed" and attempt.get("feasible_certificate")
                   and attempt.get("actual_target_squared_distance") == 0.0 for attempt in target["attempts"])
               for record in certificate["records"] for target in record["targets"]):
        return None
    return {"selected_indices": certificate["selected_indices"],
            "selected_nominal_and_perturbed_targets_attainable": True,
            "scope": "Only the two selected extreme samples; not a zero-error claim for all samples",
            "raw_evaluation_values_unchanged": True,
            "projection_records": [{"sample_index": row["sample_index"],
                                    "settings_label": row["settings_label"],
                                    "nominal_self_projection_squared_distance": row["recomputed_squared_distance"]}
                                   for row in projection["records"]],
            "evidence_sha256": {str(path): sha256(path) for path in
                                (certificate_path, projection_path, witness_path)}}


def training_config_snapshot(training_dir):
    """Read contents and mtime from one stable generation of the config file."""
    path = training_dir / "training_config.json"
    try:
        before = path.stat()
        configuration = read_json(path)
        after = path.stat()
    except FileNotFoundError:
        return None, -1
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        return None, after.st_mtime_ns
    return configuration, after.st_mtime_ns


class ProcessMonitor:
    """Read-only process handle: distinguishes a finished PID from PID reuse."""

    def __init__(self, pid, expected_creation=None):
        self.pid = int(pid)
        self.handle = None
        self.creation = None
        if self.pid <= 0:
            raise ValueError("A process ID must be positive")
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            self.ctypes = ctypes
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            self.kernel.OpenProcess.restype = wintypes.HANDLE
            self.kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            self.kernel.WaitForSingleObject.restype = wintypes.DWORD
            self.kernel.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
            self.kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            self.kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            self.handle = self.kernel.OpenProcess(0x00100000 | 0x1000, False, self.pid)
            if not self.handle:
                error = ctypes.get_last_error()
                if error == 87:  # PID no longer exists.
                    return
                raise OSError(error, f"Cannot monitor process {self.pid}")
            fields = [wintypes.FILETIME() for _ in range(4)]
            if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(value) for value in fields)):
                self.close()
                raise OSError(ctypes.get_last_error(), "Cannot read process creation time")
            self.creation = fields[0].dwLowDateTime + (fields[0].dwHighDateTime << 32)
            if expected_creation is not None and self.creation != expected_creation:
                self.close()
                raise RuntimeError(f"PID {self.pid} was reused by a different process")

    def poll(self):
        if os.name == "nt":
            if not self.handle:
                return False, None
            outcome = self.kernel.WaitForSingleObject(self.handle, 0)
            if outcome == 258:
                return True, None
            if outcome != 0:
                raise OSError(self.ctypes.get_last_error(), f"Cannot poll process {self.pid}")
            from ctypes import wintypes
            exit_code = wintypes.DWORD()
            if not self.kernel.GetExitCodeProcess(self.handle, self.ctypes.byref(exit_code)):
                raise OSError(self.ctypes.get_last_error(), "Cannot read process exit code")
            return False, int(exit_code.value)
        try:
            os.kill(self.pid, 0)
            return True, None
        except ProcessLookupError:
            return False, None

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def process_alive(pid, creation=None):
    monitor = ProcessMonitor(pid, creation)
    try:
        return monitor.poll()[0]
    finally:
        monitor.close()


def validate_training_protocol(config, *, completed=False, device_scale=None):
    if not config:
        raise RuntimeError("The training configuration is missing or incomplete")
    scale = config.get("device_scale") if device_scale is None else device_scale
    counts = device_counts(scale)
    if config.get("status") == "failed":
        raise RuntimeError(f"Training failed: {config.get('failure', 'see training_config.json')}")
    if config.get("status") == "initializing" and not completed:
        if config.get("device_scale") != scale:
            raise RuntimeError(f"The running training process is not the scale{scale} experiment")
        return
    expected = {"device_scale": scale, "device_counts": counts, "case_seed": 42, "training_seed": 42,
                "gradient_mode": "approx", "num_samples": 100, "batch_size": 5, "noise_scale": 0.05,
                "count_branching": True, "storage_headroom": True,
                "physical_solver_backend": "gurobi_persistent_integer_counts"}
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"Training protocol mismatch for {key}: {config.get(key)!r}")
    for key, value in {"Presolve": 0, "Threads": 8, "Seed": 0, "MIPFocus": 2, "Symmetry": 2}.items():
        if config.get("solver_options", {}).get(key) != value:
            raise RuntimeError(f"Training solver protocol mismatch for {key}")
    if completed and (config.get("status") != "completed" or config.get("optimizer_updates") != 2000
                      or config.get("phase_end_updates") != [600, 1200, 2000]):
        raise RuntimeError("Training did not finish all 2,000 updates and three stages")


def make_summary(config, setup, evaluation, evaluation_summary, verification, training_solver, pipeline_seconds):
    """Pure report assembly; all metric values come from finite raw statistics."""
    validate_training_protocol(config, completed=True)
    scale = config["device_scale"]
    counts = device_counts(scale)
    if (evaluation.get("status") != "completed" or evaluation.get("num_samples") != 5000
            or evaluation.get("seed") != 1 or evaluation.get("noise_scale") != 0.05
            or evaluation.get("recorded_samples") != 5000 or evaluation.get("completed_samples") != 5000
            or evaluation.get("failed_samples") != 0 or evaluation.get("device_counts") != counts
            or evaluation.get("storage_headroom") is not True):
        raise RuntimeError("Evaluation has not completed the required 5,000 samples")
    if verification.get("status") != "passed" or verification.get("independent_extremes", {}).get("status") != "passed":
        raise RuntimeError("Full verification, including independent extreme checks, did not pass")
    if verification.get("portfolio", {}).get("device_counts") != counts:
        raise RuntimeError("The verification report belongs to a different device scale")
    metrics = evaluation_summary["metrics"]
    for name in METRICS:
        if metrics[name].get("finite_count") != 5000:
            raise RuntimeError(f"Not all 5,000 raw values are finite for {name}")
        if not all(math.isfinite(float(metrics[name][key])) for key in ("mean", "median", "p95", "max")):
            raise RuntimeError(f"Nonfinite statistics for {name}")
    phases = config.get("train_calls", [])
    if len(phases) != 3 or [phase.get("optimizer_updates") for phase in phases] != [600, 600, 800]:
        raise RuntimeError("Missing full three-stage timing records")
    phase_seconds = [float(phase["elapsed_seconds"]) for phase in phases]
    timings = {"setup_seconds": float(setup["setup_seconds"]),
               "training_seconds": float(config["elapsed_seconds"]),
               "phase_seconds": phase_seconds,
               "setup_plus_training_seconds": float(setup["setup_seconds"] + config["elapsed_seconds"]),
               "evaluation_load_initialization_seconds": float(evaluation["load_initialization_seconds"]),
               "evaluation_seconds": float(evaluation["evaluation_elapsed_seconds"]),
               "evaluation_total_seconds": float(evaluation["elapsed_seconds"]),
               "pipeline_observed_seconds": float(pipeline_seconds)}
    if not all(math.isfinite(value) and value >= 0 for value in (*phase_seconds, *(v for v in timings.values() if not isinstance(v, list)))):
        raise RuntimeError("Invalid elapsed time in experiment outputs")
    return {"status": "completed", "completed_at_utc": now(), "T": 24, "device_scale": scale,
            "device_counts": counts, "model_complexity": config["model_complexity"],
            "training_seed": 42, "evaluation_seed": 1, "noise_scale": 0.05,
            "optimizer_updates": 2000, "phase_updates": [600, 600, 800],
            "evaluation_samples": 5000, "completed_samples": 5000, "failed_samples": 0,
            "timings": timings, "metrics": metrics,
            "metric_definitions": {key: value for key, value in evaluation.items() if key.endswith("_definition")},
            "solver_strategy": {"backend": config["physical_solver_backend"],
                                "physical_integer_constraints": "unrelaxed",
                                "redundant_integer_counts_per_solve": 72,
                                "redundant_count_equalities_per_solve": 72,
                                "storage_headroom": True,
                                "valid_storage_headroom_inequalities_per_solve": 48 * scale,
                                "storage_headroom_t0_skipped": False,
                                "count_branch_priority": 100, "original_branch_priority": 0,
                                "solver_options": config["solver_options"],
                                "actual_tolerances": training_solver["diagnostics"]["actual_tolerances"]},
            "verification_status": "passed", "independent_extremes_status": "passed"}


def chinese_report(summary):
    timings = summary["timings"]
    scale = summary["device_scale"]
    storage_cuts = summary["solver_strategy"]["valid_storage_headroom_inequalities_per_solve"]
    storage_scope = ("只作用于t>=1，不在t=0加约束" if summary["solver_strategy"]["storage_headroom_t0_skipped"]
                     else "覆盖全部24个功率时段，含索引0；SOC[0]为初始状态")
    labels = {
        "feasibility_error": "可行性误差（物理功率平方距离，kW²）",
        "feasibility_error_rated": "可行性误差（归一化平方距离）",
        "optimality_error_rated": "几何最优性误差（归一化平方距离）",
        "objective_error": "方向支持物理功率成本差（带符号）",
        "objective_error_rated": "方向支持相对成本差（带符号）",
    }
    lines = [f"{scale}倍 Microgrid 完整实验结果", "",
             f"场景：T=24；MT/PV/TL/CL/ESS 各{scale}台，HP {2 * scale}台，共{7 * scale}台。原整数约束全部保留。",
             "训练：seed=42，100个不确定场景，batch size=5，三个阶段共2,000次更新。",
             f"初始化及外边界计算：{timings['setup_seconds']:.3f} 秒。",
             f"训练总耗时：{timings['training_seconds']:.3f} 秒；含初始化：{timings['setup_plus_training_seconds']:.3f} 秒。"]
    for i, (updates, seconds) in enumerate(zip(summary["phase_updates"], timings["phase_seconds"]), 1):
        lines.append(f"第{i}阶段：{updates}次更新，{seconds:.3f} 秒。")
    lines.extend([f"评估：seed=1，5%随机扰动，5,000个样本全部完成，失败0个。",
                  f"评估初始化：{timings['evaluation_load_initialization_seconds']:.3f} 秒；评估计算：{timings['evaluation_seconds']:.3f} 秒；评估总耗时：{timings['evaluation_total_seconds']:.3f} 秒。", "",
                  "指标均为原始值统计，未裁剪或替换；归一化误差为平方距离之和，不是RMSE。"])
    lines.append("沿用原Microgrid评估方向c=-U(0,1)；方向支持功率成本差不是真实电价费用，不能标为EUR。")
    for name in METRICS:
        values = summary["metrics"][name]
        lines.append(f"{labels[name]}：mean={values['mean']:.9g}，median={values['median']:.9g}，p95={values['p95']:.9g}，max={values['max']:.9g}。")
    tolerances = summary["solver_strategy"]["actual_tolerances"]
    lines.extend(["", "加速策略：每次重新构造当前物理模型；新增72个等于原binary之和的冗余整数计数和72条等式，K分支优先级100、原变量0。",
                  f"另加入{storage_cuts}条由原SOC动态、SOC边界和binary互斥充放电推导的有效储能余量不等式；{storage_scope}。没有新增设备，原整数可行域保持不变。",
                  "求解设置：Presolve=0，Threads=8，Seed=0，MIPFocus=2，Symmetry=2。原可行域和目标函数保持不变。",
                  "实际默认精度：" + "，".join(f"{key}={value:g}" for key, value in tolerances.items()) + "。",
                  "校验：完整数据审查和独立极值重算均通过。方向支持成本差与几何最优性误差分别报告。"])
    review = summary.get("projection_numerical_review")
    if review:
        selected = sorted(set(review["selected_indices"].values()))
        lines.extend(["", "数值复核说明：",
                      f"极值样本{selected}的名义支持点及扰动场景近似支持点均取得固定坐标、常数目标MILP可行证书；原整数域保留，审计容差为1e-9。",
                      "默认精度的近零投影QP会产生非零残差，因此这两个样本的微小可行性误差不能解释为真实不可行；该结论只覆盖已复核样本。",
                      "表中5,000个样本的原始误差统计保持原样。名义支持点可达性使用固定点证书复核，其他独立指标重算比较容差保持原值。",
                      "几何最优性误差均值约0.4076，仍有明显的近似保守性。"])
    return "\n".join(lines) + "\n"


class Pipeline:
    def __init__(self, device_scale=3):
        self.device_scale = device_scale
        self.counts = device_counts(device_scale)
        self.output = scale_output(device_scale)
        self.training = self.output / "training"
        self.evaluation = self.output / "disaggregation"
        self.output.mkdir(parents=True, exist_ok=True)
        self.status_path = self.output / "pipeline_status.json"
        self.previous = read_json(self.status_path) or {}
        self.started = time.perf_counter()
        self.lock_path = self.output / "pipeline.lock"
        owner = read_json(self.lock_path)
        if self.lock_path.exists() and owner is None:
            raise RuntimeError("The pipeline lock is unreadable; refusing to start a duplicate supervisor")
        if owner and process_alive(owner["pid"], owner.get("creation")):
            raise RuntimeError(f"Another pipeline supervisor is still active (PID {owner['pid']})")
        if self.lock_path.exists():
            self.lock_path.unlink()
        monitor = ProcessMonitor(os.getpid())
        self.creation = monitor.creation
        monitor.close()
        with self.lock_path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "creation": self.creation}, stream)
        self.state = {"status": "running", "stage": "starting", "started_at_utc": now(),
                      "supervisor_pid": os.getpid(), "supervisor_creation": self.creation,
                      "result_dir": str(self.output), "device_scale": self.device_scale, "device_counts": self.counts}

    def update(self, stage, **details):
        self.state.update(stage=stage, updated_at_utc=now(), observed_seconds=time.perf_counter() - self.started,
                          **details)
        write_json(self.status_path, self.state)

    def wait(self, pid, stage, *, process=None, expected_creation=None, prelaunch_training_mtime_ns=None):
        monitor = ProcessMonitor(pid, expected_creation)
        fresh_training_config_seen = prelaunch_training_mtime_ns is None
        self.update(stage, **{stage + "_pid": pid, stage + "_pid_creation": monitor.creation})
        try:
            while True:
                alive, exit_code = monitor.poll()
                if process is not None:
                    return_code = process.poll()
                    if return_code is not None:
                        alive, exit_code = False, return_code
                if stage == "training":
                    config, config_mtime_ns = training_config_snapshot(self.training)
                    if (not fresh_training_config_seen and config
                            and config_mtime_ns > prelaunch_training_mtime_ns
                            and config.get("status") in ("initializing", "running", "completed", "failed")):
                        fresh_training_config_seen = True
                    if config and fresh_training_config_seen:
                        validate_training_protocol(config, device_scale=self.device_scale)
                    progress = read_json(self.training / "solver_progress.json")
                    self.update(stage, training_config_status=(config or {}).get("status"),
                                training_progress=progress, training_process_alive=alive,
                                training_config_fresh=fresh_training_config_seen,
                                training_waiting_for_new_config=not fresh_training_config_seen,
                                training_prelaunch_config_mtime_ns=prelaunch_training_mtime_ns)
                elif stage == "evaluation":
                    self.update(stage, evaluation_progress=read_json(self.evaluation / "metadata.json"),
                                evaluation_process_alive=alive)
                else:
                    self.update(stage, verification_process_alive=alive)
                if not alive:
                    if exit_code not in (None, 0):
                        raise RuntimeError(f"{stage} process {pid} exited with code {exit_code}")
                    if stage == "training" and not fresh_training_config_seen:
                        raise RuntimeError("Training process exited without publishing a fresh training configuration")
                    break
                # Imports may take time before the runner publishes initializing.
                # Watch the short startup transition without accepting old output.
                time.sleep(0.1 if stage == "training" and not fresh_training_config_seen else 10)
        finally:
            monitor.close()
        if stage == "training":
            config = read_json(self.training / "training_config.json")
            validate_training_protocol(config, completed=True, device_scale=self.device_scale)
            missing = [name for name in FINAL_TRAINING_FILES if not (self.training / name).is_file()]
            if missing:
                raise RuntimeError(f"Training exited before final artifacts were written: {missing}")

    def launch(self, stage, command):
        environment = os.environ.copy()
        environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", KMP_DUPLICATE_LIB_OK="TRUE")
        stdout_path, stderr_path = (self.output / f"{stage}_{suffix}.log" for suffix in ("stdout", "stderr"))
        prelaunch_training_mtime_ns = None
        if stage == "training":
            try:
                prelaunch_training_mtime_ns = (self.training / "training_config.json").stat().st_mtime_ns
            except FileNotFoundError:
                prelaunch_training_mtime_ns = -1
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(command, cwd=CODE, env=environment, stdout=stdout, stderr=stderr,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.update(stage, command=command, **{stage + "_stdout_log": str(stdout_path),
                                                   stage + "_stderr_log": str(stderr_path)})
            self.wait(process.pid, stage, process=process,
                      prelaunch_training_mtime_ns=prelaunch_training_mtime_ns)

    def resume_child(self, stage):
        if self.previous.get("stage") != stage:
            return False
        pid = self.previous.get(stage + "_pid")
        creation = self.previous.get(stage + "_pid_creation")
        if not pid or not process_alive(pid, creation):
            return False
        self.update(stage, adopted_child=True)
        self.wait(pid, stage, expected_creation=creation)
        return True

    def evaluation_reusable(self):
        metadata = read_json(self.evaluation / "metadata.json") or {}
        if (metadata.get("status") != "completed" or metadata.get("num_samples") != 5000
                or metadata.get("seed") != 1 or metadata.get("noise_scale") != .05
                or metadata.get("recorded_samples") != 5000 or metadata.get("completed_samples") != 5000
                or metadata.get("failed_samples") != 0 or metadata.get("device_counts") != self.counts
                or metadata.get("T") != 24 or metadata.get("count_branching") is not True
                or metadata.get("storage_headroom") is not True
                or metadata.get("backend") != "gurobi_persistent_integer_counts"):
            return False
        if not all((self.evaluation / name).is_file() for name in ("summary.json", "results.pkl", "errors.csv", "solver_diagnostics.json")):
            return False
        hashes = metadata.get("training_source_sha256", {})
        required = [self.training / name for name in ("metadata.pkl", "pretrainnet_weights.pth", "final_polytope.npz", "training_config.json")]
        return all(hashes.get(str(path.resolve())) == sha256(path) for path in required)

    def verification_reusable(self):
        report = read_json(self.output / "verification.json") or {}
        hashes = report.get("verified_artifact_sha256", {})
        return (report.get("status") == "passed" and report.get("independent_extremes", {}).get("status") == "passed"
                and report.get("portfolio", {}).get("device_counts") == self.counts
                and bool(hashes) and all((ROOT / name).is_file() and sha256(ROOT / name) == expected
                                         for name, expected in hashes.items()))

    def run(self, waiting_pid):
        for name in ("summary.json", "report.txt"):
            path = self.output / name
            if path.is_file():
                path.unlink()
        if waiting_pid is not None:
            config = read_json(self.training / "training_config.json")
            validate_training_protocol(config, device_scale=self.device_scale)
            expected = (self.previous.get("training_pid_creation")
                        if self.previous.get("training_pid") == waiting_pid else None)
            self.update("training", adopted_training=True,
                        training_stdout_log="External training stdout remains with its original launcher")
            self.wait(waiting_pid, "training", expected_creation=expected)
        else:
            config = read_json(self.training / "training_config.json") or {}
            if config.get("status") in ("initializing", "running"):
                raise RuntimeError("Training is already marked active; use --wait-training-pid to adopt it instead of starting another training")
            self.launch("training", [sys.executable, "-u", "-m", "Simulator.runners.train_microgrid",
                                      "--device-scale", str(self.device_scale), "--mode", "approx", "--case-seed", "42",
                                      "--training-seed", "42", "--solver-threads", "8", "--mip-focus", "2",
                                      "--solver-symmetry", "2", "--count-branching", "--storage-headroom", "--overwrite"])
        if not self.resume_child("evaluation") and not self.evaluation_reusable():
            self.launch("evaluation", [sys.executable, "-u", "-m", "Simulator.testers.evaluate_microgrid_disaggregation",
                                        "--training-dir", str(self.training), "--output-dir", str(self.evaluation),
                                        "--num-samples", "5000", "--seed", "1", "--noise-scale", "0.05",
                                        "--count-branching", "--storage-headroom", "--solver-threads", "8", "--overwrite"])
        if not VERIFIER.is_file():
            raise FileNotFoundError(VERIFIER)
        if not self.resume_child("verification") and not self.verification_reusable():
            path = self.output / "verification.json"
            if path.is_file():
                path.unlink()
            self.launch("verification", [sys.executable, "-u", str(VERIFIER), "--device-scale", str(self.device_scale), "--result-dir", str(self.output)])
        self.update("reporting")
        summary = make_summary(read_json(self.training / "training_config.json"), read_json(self.training / "setup_timing.json"),
                               read_json(self.evaluation / "metadata.json"), read_json(self.evaluation / "summary.json"),
                               read_json(self.output / "verification.json"), read_json(self.training / "solver_diagnostics.json"),
                               time.perf_counter() - self.started)
        review = numerical_review(self.output)
        if review:
            summary["projection_numerical_review"] = review
        write_json(self.output / "summary.json", summary)
        (self.output / "report.txt").write_text(chinese_report(summary), encoding="utf-8")
        self.update("completed", status="completed", summary=str(self.output / "summary.json"),
                    report=str(self.output / "report.txt"))

    def close(self):
        owner = read_json(self.lock_path)
        if owner and owner.get("pid") == os.getpid() and owner.get("creation") == self.creation:
            self.lock_path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-scale", type=int, choices=(3, 5), default=3,
                        help="Retained portfolio scale (default: 3)")
    parser.add_argument("--wait-training-pid", type=int,
                        help="Adopt this existing training PID, without launching a second training")
    args = parser.parse_args(argv)
    if args.wait_training_pid is not None and args.wait_training_pid <= 0:
        parser.error("--wait-training-pid must be positive")
    pipeline = Pipeline(args.device_scale)
    try:
        pipeline.run(args.wait_training_pid)
    except Exception as exc:
        pipeline.update("failed", status="failed", failure=f"{type(exc).__name__}: {exc}",
                        traceback=traceback.format_exc())
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
