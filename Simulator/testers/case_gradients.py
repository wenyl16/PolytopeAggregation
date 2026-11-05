"""Paired exact/approximate gradient case: training, evaluation, checks and summaries."""

# Paired training and fitting.
import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from Simulator.Approximator import ErrorCalculator, PreTrainNet, compute_exact_error, compute_surrogate_error
from Simulator.cases.aggregation_case_T24 import Aggregator
from Simulator.paths import gradient_result_dir

STAGES = [(300, 0.02, 1.0), (200, 0.1, 0.01), (500, 1.0, 1e-6)]
OPTIONS = {'FeasibilityTol': 1e-9, 'OptimalityTol': 1e-9,
           'BarConvTol': 1e-12, 'Threads': 1, 'Seed': 0}
DEFAULT_TRAINING_SEEDS = tuple(range(0, 1000, 100))

def surrogate(A, b, samples):
    return compute_surrogate_error(A, b, [samples])

def configure(calculator):
    calculator.configure(feas_tol=0., opt_tol=0., active_tol=1e-5)
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(OPTIONS)
    return calculator



def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding='utf-8')


def sample_errors(calculator, direction, exact=False):
    """Identical four LP/QP calls; only the derivative construction differs."""
    samples = []
    solve_seconds = 0.
    analytic_seconds = 0.
    for kind in ('feas', 'opt'):
        start = time.perf_counter()
        if kind == 'feas':
            x_apx = calculator.optimize_direction(direction, in_approx=True)
            x_org = calculator.project(x_apx) if x_apx is not None else None
        else:
            x_org = calculator.optimize_direction(direction)
            x_apx = calculator.project(x_org, to_approx=True) if x_org is not None else None
        solve_seconds += time.perf_counter() - start
        if x_apx is None or x_org is None:
            raise RuntimeError(f'{kind}: LP/QP solve failed')
        start = time.perf_counter()
        if exact:
            result = calculator._error_result(x_apx, x_org, kind, direction)
            if not result['is_valid']:
                raise RuntimeError('Exact gradient failed certification')
        else:
            result = {'is_valid': True,
                      'error': float(np.sum((x_apx - x_org) ** 2)),
                      'active_indices': calculator._find_active(x_apx),
                      'x_org': x_org.copy(), 'x_apx': x_apx.copy()}
        analytic_seconds += time.perf_counter() - start
        samples.append([result])
    return samples, solve_seconds, analytic_seconds


def make_optimizer(network, lr):
    optimizer = torch.optim.SGD(network.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
    return optimizer, scheduler


def take_step(run, direction, iteration, stage, weight, device):
    sync(device)
    started = time.perf_counter()
    run['optimizer'].zero_grad(set_to_none=True)
    A, b = run['network']()
    A_numpy = A[0].detach().cpu().numpy().copy()
    b_numpy = b[0].detach().cpu().numpy().copy()
    run['calculator'].update_polytope(A_hat=A_numpy, b_hat=b_numpy)
    sync(device)
    prepare_seconds = time.perf_counter() - started
    samples, solve_seconds, analytic_seconds = sample_errors(
        run['calculator'], direction, exact=run['method'] == 'exact')
    start = time.perf_counter()
    if run['method'] == 'exact':
        feas = compute_exact_error(A, b, [samples[0]])
        opt = compute_exact_error(A, b, [samples[1]])
    else:
        feas = surrogate(A, b, samples[0])
        opt = surrogate(A, b, samples[1])
    (feas + weight * opt).backward()
    gradient_norm = torch.sqrt(sum(p.grad.square().sum() for p in run['network'].parameters()
                                   if p.grad is not None)).item()
    lr = run['optimizer'].param_groups[0]['lr']
    run['optimizer'].step()
    run['scheduler'].step()
    sync(device)
    backward_seconds = time.perf_counter() - start
    row = {'method': run['method'], 'iteration': iteration, 'stage': stage,
           'lr': lr, 'rate_opt_feas': weight,
           'feasibility_error': samples[0][0]['error'],
           'optimality_error': samples[1][0]['error'],
           'objective_feasibility': feas.item(), 'objective_optimality': opt.item(),
           'gradient_norm': gradient_norm,
           'step_seconds': time.perf_counter() - started,
           'solve_seconds': solve_seconds, 'analytic_gradient_seconds': analytic_seconds,
           'backward_update_seconds': backward_seconds, 'prepare_seconds': prepare_seconds}
    for kind, sample in zip(('feasibility', 'optimality'), samples):
        sample = sample[0]
        row[f'{kind}_gradient_status'] = sample.get('gradient_status', 'surrogate')
        row[f'{kind}_adjoint_solver'] = sample.get('adjoint_solver', '')
        row[f'{kind}_active_count'] = len(sample['active_indices'])
        row[f'{kind}_gradient_active_count'] = len(sample.get('gradient_active_indices', sample['active_indices']))
        for diagnostic in ('active_matrix_rank', 'active_matrix_condition',
                           'adjoint_residual_inf', 'active_constraint_residual_inf',
                           'linearization_incompatibility_dimension'):
            row[f'{kind}_{diagnostic}'] = sample.get(diagnostic, '')
    run['rows'].append(row)
    run['used_directions'].append(direction.copy())
    return row, (A_numpy, b_numpy, samples)


def stats(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'median': float(np.median(a)),
            'max': float(a.max()), 'min': float(a.min()),
            'p95': float(np.quantile(a, .95)), 'last': float(a[-1])}


def gradient_comparison(snapshot, output_dir):
    """Compare both gradients at the SAME A,b,points,active sets, and v."""
    iteration, A, b, samples, direction = snapshot
    output = []
    arrays = {'A': A, 'b': b, 'direction': direction}
    for kind, sample in zip(('feasibility', 'optimality'), samples):
        sample = sample[0]
        At = torch.tensor(A[None], dtype=torch.float64, requires_grad=True)
        bt = torch.tensor(b[None], dtype=torch.float64, requires_grad=True)
        value = surrogate(At, bt, [sample])
        gA, gb = torch.autograd.grad(value, (At, bt))
        approximate = np.concatenate((gA.numpy().ravel(), gb.numpy().ravel()))
        exact = np.concatenate((sample['gradient_A'].ravel(), sample['gradient_b']))
        ne, na = np.linalg.norm(exact), np.linalg.norm(approximate)
        output.append({'iteration': iteration, 'kind': kind, 'true_error': sample['error'],
                       'gradient_status': sample.get('gradient_status', 'ok'),
                       'surrogate_value': value.item(), 'exact_norm': float(ne),
                       'approximate_norm': float(na),
                       'relative_gradient_difference': float(np.linalg.norm(approximate-exact)/ne)
                       if ne > 1e-12 else None,
                       'cosine': float(np.dot(exact, approximate)/(ne*na))
                       if min(ne, na) > 1e-12 else None})
        for name in ('gradient_A', 'gradient_b', 'x_org', 'x_apx', 'active_indices'):
            arrays[f'{kind}_{name}'] = sample[name]
        arrays[f'{kind}_gradient_status'] = sample.get('gradient_status', 'ok')
        if 'adjoint_solver' in sample:
            arrays[f'{kind}_adjoint_solver'] = sample['adjoint_solver']
        if 'gradient_active_indices' in sample:
            arrays[f'{kind}_gradient_active_indices'] = sample['gradient_active_indices']
        for diagnostic in ('active_matrix_rank', 'active_matrix_condition',
                           'adjoint_residual_inf', 'active_constraint_residual_inf',
                           'linearization_incompatibility_dimension'):
            if diagnostic in sample:
                arrays[f'{kind}_{diagnostic}'] = sample[diagnostic]
        arrays[f'{kind}_surrogate_gradient_A'] = gA.numpy()[0]
        arrays[f'{kind}_surrogate_gradient_b'] = gb.numpy()[0]
    np.savez(output_dir / f'same_point_{iteration:04d}.npz', **arrays)
    return output


def train_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=1000, help='Actual updates per method, 1..1000')
    parser.add_argument('--eval-directions', type=int, default=128, help='Independent reference prefix saved after training')
    parser.add_argument('--samples', type=int, default=10000, help='Final evaluation directions')
    parser.add_argument('--seed', type=int, default=0, help='Case and initialization seed')
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument('--training-seed', type=int, help='Run one seed')
    seeds.add_argument('--training-seeds', type=int, nargs='+',
                       help='Run a list; default: 0 100 200 300 400 500 600 700 800 900')
    parser.add_argument('--evaluation-seed', type=int, default=10000)
    parser.add_argument('--reference-run', type=Path, help='Optional comparison with an existing complete run')
    parser.add_argument('--output', type=Path, help='New run directory, or new batch parent for multiple seeds')
    parser.add_argument('--train-only', action='store_true', help='Skip final evaluation and recheck')
    parser.add_argument('--overwrite', action=argparse.BooleanOptionalAction, default=True,
                        help='Replace this identified fixed paired experiment in place (default: enabled)')
    args = parser.parse_args(argv)
    if not 1 <= args.iterations <= 1000 or args.eval_directions < 1 or args.samples < 1:
        parser.error('Use 1..1000 updates and positive evaluation sample counts')
    selected = ([args.training_seed] if args.training_seed is not None else
                args.training_seeds if args.training_seeds is not None else list(DEFAULT_TRAINING_SEEDS))
    if len(set(selected)) != len(selected):
        parser.error('Training seeds must be distinct')
    multiple = len(selected) > 1
    cohort = gradient_result_dir() / 'run_20261019_090229_case0_seeds0_900'
    base = (args.output or (cohort if multiple else cohort / f'seed_{selected[0]}')).resolve()
    if base.exists() and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite: {base}')
    if multiple:
        base.mkdir(parents=True, exist_ok=args.overwrite)
    status = {'status': 'running', 'training_seeds': selected, 'case_seed': args.seed,
              'actual_updates': args.iterations, 'evaluation_samples': args.samples,
              'evaluation_seed': args.evaluation_seed, 'train_only': args.train_only,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'runs': []}
    completed = []
    for seed in selected:
        run_args = copy.copy(args)
        run_args.training_seed = seed
        run_args.output = base / f'seed_{seed}' if multiple else base
        record = {'training_seed': seed, 'path': run_args.output.name, 'status': 'running'}
        status['runs'].append(record)
        phase = 'training'
        try:
            output = run_pair(run_args)
            if not args.train_only:
                phase = 'evaluation'
                evaluation = evaluate_final(argparse.Namespace(result_dir=output, samples=args.samples, seed=args.evaluation_seed))
                phase = 'recheck'
                recheck_extremes(argparse.Namespace(evaluation_dir=evaluation))
            record['status'] = 'completed'
            completed.append(output)
        except Exception as exc:
            record.update(status='failed', phase=phase, error=f'{type(exc).__name__}: {exc}')
            print(f'Seed {seed} failed during {phase}: {exc}', flush=True)
        record['ended_utc'] = datetime.now(timezone.utc).isoformat()
        if base.exists():
            write_json(base / 'batch_status.json', status)
    status['status'] = 'completed' if len(completed) == len(selected) else 'completed_with_failures'
    status['ended_utc'] = datetime.now(timezone.utc).isoformat()
    if multiple and not args.train_only:
        summarize_runs([base / f'seed_{seed}' for seed in selected], base / 'summary',
                       samples=args.samples, seed=args.evaluation_seed)
    if base.exists():
        write_json(base / 'batch_status.json', status)
    print(f'Batch {status["status"]}: {base}', flush=True)
    return 0 if status['status'] == 'completed' else 1


def run_pair(args):
    import gurobipy
    training_seed = args.seed if args.training_seed is None else args.training_seed
    evaluation_seed = args.seed + 10000 if args.evaluation_seed is None else args.evaluation_seed
    if not (1 <= args.iterations <= 1000) or args.eval_directions < 1:
        raise ValueError('Use 1..1000 iterations and at least one evaluation direction')
    root = gradient_result_dir()
    output = args.output or root / 'run_20261019_090229_case0_seeds0_900' / f'seed_{training_seed}'
    if output.exists() and getattr(args, 'overwrite', False):
        target = output.resolve()
        if (output.is_symlink() or not target.is_relative_to(root.resolve())
                or target == root.resolve()):
            raise ValueError('Overwrite must identify one paired result inside gradient_comparison')
        marker = target / 'metadata.json'
        old = json.loads(marker.read_text(encoding='utf-8'))
        if old.get('model') != 'PreTrainNet' or old.get('seed') != args.seed or old.get('training_seed', old.get('seed')) != training_seed:
            raise ValueError('Refusing to overwrite an unidentified or differently seeded pair')
        for path in target.rglob('*'):
            if path.is_file():
                if path.is_symlink() or not path.resolve().is_relative_to(target):
                    raise ValueError(f'Unsafe paired output: {path}')
                path.unlink()
    output.mkdir(parents=True, exist_ok=getattr(args, 'overwrite', False))
    print(f'Output: {output}', flush=True)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    directions = np.random.RandomState(training_seed).randn(args.iterations, 24)
    eval_directions = np.random.RandomState(evaluation_seed).randn(args.eval_directions, 24)
    np.save(output / 'training_directions.npy', directions)
    np.save(output / 'evaluation_directions.npy', eval_directions)
    agg = Aggregator(seed=args.seed, discrete_rate=0.)
    agg.gen_CL(5)
    agg.gen_TCL(40)
    agg.gen_PV(5)
    original = {'model': agg.case_aggregator(build_only=True)}
    cumulative = np.tril(np.ones((24, 24)))
    cumulative /= cumulative.sum(axis=1, keepdims=True)
    initial_A = np.vstack((np.eye(24), -np.eye(24), cumulative[1:], -cumulative[1:]))
    initializer = configure(ErrorCalculator(original, initial_A.copy(), np.zeros(94)))
    initial_b = np.array([row @ initializer.optimize_direction(-row) for row in initial_A])
    template = PreTrainNet(initial_A, initial_b, device=device).double().to(device)
    state = copy.deepcopy(template.state_dict())
    with torch.no_grad():
        initial_normalized_A, initial_normalized_b = template()
    np.savez(output / 'initial_polytope.npz',
             A=initial_normalized_A[0].cpu().numpy(), b=initial_normalized_b[0].cpu().numpy())
    reference_check = None
    if args.reference_run is not None:
        reference_meta = json.loads((args.reference_run / 'metadata.json').read_text(encoding='utf-8'))
        assert reference_meta['seed'] == args.seed, 'Reference device-case seed differs'
        with np.load(args.reference_run / 'initial_polytope.npz') as reference_initial:
            np.testing.assert_array_equal(initial_normalized_A[0].cpu().numpy(), reference_initial['A'])
            np.testing.assert_array_equal(initial_normalized_b[0].cpu().numpy(), reference_initial['b'])
        reference_eval = np.load(args.reference_run / 'evaluation_directions.npy')
        prefix = min(len(reference_eval), len(eval_directions))
        np.testing.assert_array_equal(eval_directions[:prefix], reference_eval[:prefix])
        reference_train = np.load(args.reference_run / 'training_directions.npy')
        train_prefix = min(len(reference_train), len(directions))
        training_changed = not np.array_equal(directions[:train_prefix], reference_train[:train_prefix])
        if training_seed != reference_meta.get('training_seed', reference_meta['seed']):
            assert training_changed, 'Requested new seed produced unchanged training directions'
        reference_check = {'run': str(args.reference_run.resolve()),
                           'initial_A_and_b_exactly_equal': True,
                           'evaluation_prefix_exactly_equal': True,
                           'evaluation_prefix_length': prefix,
                           'training_directions_changed': training_changed}
    print(f'Seeds: case/initialization={args.seed}, training={training_seed}, '
          f'evaluation={evaluation_seed}; reference check={reference_check}', flush=True)
    metadata = {'status': 'running', 'seed': args.seed, 'case_seed': args.seed,
                'initialization_seed': args.seed, 'training_seed': training_seed,
                'evaluation_seed': evaluation_seed, 'reference_check': reference_check,
                'iterations': args.iterations, 'evaluation_directions': args.eval_directions,
                'python': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
                'gurobi': list(gurobipy.gurobi.version()), 'device': str(device),
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                'dtype': 'float64', 'model': 'PreTrainNet', 'polytope_shape': [94, 24],
                'stages': STAGES, 'solver_options': OPTIONS, 'active_tol': 1e-5,
                'error_cutoff_both_methods': 0., 'surrogate_implementation': 'shared Approximator.compute_surrogate_error',
                'gradient_mode': 'exact_vs_approx', 'active_basis_policy': 'qr',
                'gradient_policy': 'Exactly T active rows: direct solve; more than T: thin QR '
                                   'minimum-norm sensitivity; full Gurobi projection duals; no fallback. '
                                   'Nonregular vertices are labelled sensitivities, not certified ordinary derivatives.',
                'training_directions_sha256': hashlib.sha256(directions.tobytes()).hexdigest(),
                'iteration_order': 'alternate approximate/exact first each iteration; serial execution',
                'timing_note': 'Sum of synchronized timed steps; excludes setup/warmup/logging/saving/evaluation. '
                               'Solver time includes Pyomo solve-call overhead. Single paired run.',
                'initialization_note': 'Identical production PreTrainNet initialization (biases created '
                                       'as float32 then promoted to float64), same normalization epsilon 1e-8.',
                'benchmark_adaptations': 'Shared network/loop; dtype-safe unchanged surrogate formula; '
                                         'zero cutoff both methods; no callbacks; identical saved v; '
                                         '1000 rather than legacy 1003 updates.',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), Path(__file__).parents[1]/'Approximator.py',
                    Path(__file__).parents[1]/'cases/aggregation.py',
                    Path(__file__).parents[1]/'cases/aggregation_case_T24.py')}}
    if reference_check is not None:
        comparable_keys = ('iterations', 'stages', 'solver_options', 'active_tol',
                           'error_cutoff_both_methods', 'dtype', 'model', 'device',
                           'polytope_shape', 'initialization_note')
        # JSON normalizes stage tuples to lists in the saved reference metadata.
        normalized_metadata = json.loads(json.dumps(metadata))
        for key in comparable_keys:
            assert normalized_metadata[key] == reference_meta[key], f'Reference setting differs: {key}'
        reference_check['matching_settings'] = list(comparable_keys)
        reference_check['algorithm_sources_match'] = {
            name: metadata['source_sha256'][name] == reference_meta['source_sha256'][name]
            for name in metadata['source_sha256'] if name in reference_meta['source_sha256']}
        reference_check['production_algorithm_sources_unchanged'] = all(
            reference_check['algorithm_sources_match'].values())
        reference_check['metadata_sha256'] = hashlib.sha256(
            (args.reference_run / 'metadata.json').read_bytes()).hexdigest()
    write_json(output / 'metadata.json', metadata)
    runs = {}
    for method in ('approximate', 'exact'):
        network = copy.deepcopy(template)
        network.load_state_dict(state)
        calculator = configure(ErrorCalculator(
            original, initial_A.copy(), initial_b.copy(),
            gradient_mode='exact' if method == 'exact' else 'approx'))
        run = {'method': method, 'network': network, 'calculator': calculator,
               'rows': [], 'used_directions': []}
        optimizer, scheduler = make_optimizer(network, STAGES[0][1])
        run.update(optimizer=optimizer, scheduler=scheduler)
        # Untimed warmup: solver/license initialization and CUDA backward startup.
        A, b = network()
        calculator.update_polytope(A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy())
        try:
            warmup, _, _ = sample_errors(calculator, directions[0], exact=method == 'exact')
        except Exception as exc:
            metadata.update(status='failed', failed_method=method, failure_phase='warmup',
                            failed_direction_index=1, completed_paired_iterations=0,
                            failure_type=type(exc).__name__, failure_message=str(exc))
            np.savez(output / 'failure_input.npz', A=calculator.A_hat,
                     b=calculator.b_hat, direction=directions[0])
            write_json(output / 'metadata.json', metadata)
            raise
        if method == 'exact':
            warmup_loss = compute_exact_error(A, b, [warmup[0]])
        else:
            warmup_loss = surrogate(A, b, warmup[0])
        warmup_loss.backward()
        network.zero_grad(set_to_none=True)
        sync(device)
        runs[method] = run
    for p, q in zip(runs['exact']['network'].parameters(), runs['approximate']['network'].parameters()):
        assert torch.equal(p, q), 'Initial network parameters differ'
    metadata['exact_gradient_configuration'] = runs['exact']['calculator'].params.copy()
    write_json(output / 'metadata.json', metadata)
    snapshots = []
    current_stage = None
    with (output / 'training.csv').open('w', encoding='utf-8', newline='') as f:
        writer = None
        for index, direction in enumerate(directions):
            stage = 1 if index < 300 else (2 if index < 500 else 3)
            _, lr, weight = STAGES[stage-1]
            if stage != current_stage:
                for run in runs.values():
                    run['optimizer'], run['scheduler'] = make_optimizer(run['network'], lr)
                current_stage = stage
            order = ('approximate', 'exact') if index % 2 == 0 else ('exact', 'approximate')
            for method in order:
                try:
                    row, details = take_step(runs[method], direction, index+1, stage, weight, device)
                except Exception as exc:
                    metadata.update(status='failed', failed_method=method,
                                    failure_phase='training', failed_iteration=index+1,
                                    failed_direction_index=index+1,
                                    completed_paired_iterations=index,
                                    failure_type=type(exc).__name__, failure_message=str(exc))
                    calculator = runs[method]['calculator']
                    np.savez(output / 'failure_input.npz', A=calculator.A_hat,
                             b=calculator.b_hat, direction=direction)
                    write_json(output / 'metadata.json', metadata)
                    raise
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                if method == 'exact' and index+1 in {1, 100, 300, 500, args.iterations}:
                    snapshots.append((index+1, *details, direction.copy()))
            if index == 0:
                for key in ('feasibility_error', 'optimality_error'):
                    np.testing.assert_allclose(runs['exact']['rows'][0][key],
                                               runs['approximate']['rows'][0][key], rtol=1e-7, atol=1e-10)
            if (index+1) % 50 == 0 or index+1 == args.iterations:
                f.flush()
                print(f'Paired iteration {index+1}/{args.iterations}: ' + ' | '.join(
                    f'{name}: feas={run["rows"][-1]["feasibility_error"]:.3e}, '
                    f'opt={run["rows"][-1]["optimality_error"]:.3e}, '
                    f'time={sum(r["step_seconds"] for r in run["rows"]):.2f}s'
                    for name, run in runs.items()), flush=True)
    gradient_results = []
    for snapshot in snapshots:
        gradient_results.extend(gradient_comparison(snapshot, output))
    write_json(output / 'same_point_gradients.json', gradient_results)
    summary = {'metadata': metadata, 'training': {}, 'evaluation': {},
               'same_point_gradients': gradient_results}
    for method, run in runs.items():
        used = np.array(run['used_directions'])
        assert np.array_equal(used, directions), 'Sampled directions differ'
        np.save(output / f'{method}_used_directions.npy', used)
        network = run['network']
        with torch.no_grad():
            A, b = network()
        A, b = A[0].cpu().numpy(), b[0].cpu().numpy()
        np.savez(output / f'{method}_final.npz', A=A, b=b)
        torch.save(network.state_dict(), output / f'{method}_weights.pth')
        rows = run['rows']
        summary['training'][method] = {
            key: float(sum(r[key] for r in rows)) for key in (
                'step_seconds', 'solve_seconds', 'analytic_gradient_seconds',
                'backward_update_seconds', 'prepare_seconds')}
        for key in ('feasibility_error', 'optimality_error'):
            summary['training'][method][key] = stats([r[key] for r in rows])
            summary['training'][method][f'{key}_last100_mean'] = float(np.mean([r[key] for r in rows[-100:]]))
        summary['training'][method]['gradient_status_counts'] = {
            kind: {status: sum(r[f'{kind}_gradient_status'] == status for r in rows)
                   for status in sorted({r[f'{kind}_gradient_status'] for r in rows})}
            for kind in ('feasibility', 'optimality')}
        summary['training'][method]['active_set_diagnostics'] = {
            kind: {
                diagnostic: max((r[f'{kind}_{diagnostic}'] for r in rows
                                 if r[f'{kind}_{diagnostic}'] != ''), default=None)
                for diagnostic in ('active_matrix_rank', 'active_matrix_condition',
                                   'adjoint_residual_inf', 'active_constraint_residual_inf',
                                   'linearization_incompatibility_dimension')}
            for kind in ('feasibility', 'optimality')}
        # Common evaluator uses distances only; no gradient/active-set certification is required.
        evaluator = configure(ErrorCalculator(original, A, b))
        evaluation = []
        for i, direction in enumerate(eval_directions):
            samples, _, _ = sample_errors(evaluator, direction)
            evaluation.append((i+1, samples[0][0]['error'], samples[1][0]['error']))
        with (output / f'{method}_evaluation.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['direction_index', 'feasibility_error', 'optimality_error'])
            writer.writerows(evaluation)
        summary['evaluation'][method] = {
            key: stats([row[j] for row in evaluation])
            for j, key in ((1, 'feasibility_error'), (2, 'optimality_error'))}
        print(f'{method}: training={summary["training"][method]["step_seconds"]:.3f}s; '
              f'evaluation={summary["evaluation"][method]}', flush=True)
    metadata['status'] = 'completed'
    metadata['directions_identical_verified'] = True
    summary['result_note'] = ('All compared errors are real squared distances, not surrogate loss. '
                             'Evaluation is on a shared independent sample of directions, not a worst-case certificate.')
    write_json(output / 'metadata.json', metadata)
    write_json(output / 'summary.json', summary)
    print(f'Completed: {output}', flush=True)
    return output





# Saved-result evaluation, finite-difference checks and aggregation.

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pyomo.environ as pyo
from Simulator.Approximator import ErrorCalculator

EVALUATION_OPTIONS = {'FeasibilityTol': 1e-9, 'OptimalityTol': 1e-9,
           'BarConvTol': 1e-12, 'Threads': 1, 'Seed': 0}
METHODS = ('approximate', 'exact')
KINDS = ('feasibility', 'optimality')
FEASIBILITY_COUNT_THRESHOLD = 1e-4
ERROR_THRESHOLD = 1e-8  # finite-difference resolvability, not a direction-count threshold
ACTIVE_TOL = 1e-5



def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_evaluation_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding='utf-8')


class NativeDomain:
    """Import the unchanged Pyomo constraints once, then replace objectives."""

    def __init__(self, source):
        import gurobipy as gp
        for objective in source.component_data_objects(pyo.Objective, active=True):
            objective.deactivate()
        self.source = source
        self.bridge = pyo.SolverFactory('gurobi_persistent')
        self.bridge.set_instance(source)
        self.model = self.bridge._solver_model
        self.model.setParam('OutputFlag', 0)
        for key, value in EVALUATION_OPTIONS.items():
            self.model.setParam(key, value)
        self.x = [self.bridge._pyomo_var_to_solver_var_map[source.var_proj[j]]
                  for j in range(len(source.var_proj))]
        self.model.update()
        # Pyomo's aggregate coordinates are free; do not introduce lb=0.
        if any(v.LB > -gp.GRB.INFINITY or v.UB < gp.GRB.INFINITY for v in self.x):
            raise RuntimeError('Unexpected bounds on aggregate coordinates')
        if self.model.IsMIP:
            raise RuntimeError('This evaluation is for the shared continuous case')
        self.maximum_violation = 0.
        self.solve_count = 0

    def solve(self, objective):
        import gurobipy as gp
        self.model.setObjective(objective, gp.GRB.MINIMIZE)
        self.model.reset(0)
        self.model.optimize()
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f'Gurobi status {self.model.Status}, not OPTIMAL')
        result = np.asarray(self.model.getAttr('X', self.x), dtype=float)
        violation = max(self.model.ConstrVio, self.model.BoundVio)
        if not np.all(np.isfinite(result)) or violation > 1e-7:
            raise RuntimeError(f'Invalid solve: max constraint/bound violation {violation}')
        self.maximum_violation = max(self.maximum_violation, violation)
        self.solve_count += 1
        return result, float(violation)

    def support(self, direction):
        import gurobipy as gp
        return self.solve(gp.LinExpr(direction.tolist(), self.x))

    def project(self, target):
        # Keep the constant term too, matching the full-square original QP.
        import gurobipy as gp
        return self.solve(gp.quicksum((x - float(t)) * (x - float(t))
                                     for x, t in zip(self.x, target)))


def statistics(values, *, feasibility=False):
    a = np.asarray(values, dtype=float)
    result = {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
              'p95': float(np.quantile(a, .95)), 'p99': float(np.quantile(a, .99)),
              'max': float(a.max()), 'min': float(a.min()),
              'max_direction_index_one_based': int(a.argmax()+1)}
    if feasibility:
        result['count_gt_1e-4'] = int(np.count_nonzero(a > FEASIBILITY_COUNT_THRESHOLD))
    return result


def validate_prefix(rows, references):
    validation = {}
    for method in METHODS:
        validation[method] = {}
        for kind in ('feasibility', 'optimality'):
            actual = np.array([r[f'{method}_{kind}_error'] for r in rows])
            expected = np.array([float(r[f'{kind}_error']) for r in references[method][:len(rows)]])
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9,
                                       err_msg=f'Persistent evaluator differs: {method}/{kind}')
            validation[method][kind] = {'max_absolute_difference': float(np.max(abs(actual-expected))),
                                         'rtol': 1e-6, 'atol': 1e-9}
    return {'passed': True, 'directions_checked': len(rows), 'differences': validation}


def reference_errors(calculator, directions):
    """Independent prefix check through the original per-call solver interface."""
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(EVALUATION_OPTIONS)
    rows = []
    for index, direction in enumerate(directions):
        x_apx = calculator.optimize_direction(direction, in_approx=True)
        q_org = calculator.project(x_apx) if x_apx is not None else None
        x_org = calculator.optimize_direction(direction)
        q_apx = calculator.project(x_org, to_approx=True) if x_org is not None else None
        if any(point is None for point in (x_apx, q_org, x_org, q_apx)):
            raise RuntimeError('Reference LP/QP solve failed')
        rows.append({'direction_index': index+1,
                     'feasibility_error': float(np.sum((x_apx-q_org)**2)),
                     'optimality_error': float(np.sum((x_org-q_apx)**2))})
    return rows


def evaluate_final(args):
    import gurobipy as gp
    if args.samples < 1:
        raise ValueError('At least one evaluation direction is required')
    root = args.result_dir.resolve()
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if meta['status'] != 'completed':
        raise RuntimeError('Training is not complete')
    output = root / f'evaluation_{args.samples}_seed{args.seed}'
    output.mkdir(exist_ok=False)
    directions = np.random.RandomState(args.seed).randn(args.samples, 24)
    old_directions = np.load(root/'evaluation_directions.npy', allow_pickle=False)
    common_prefix = min(len(directions), len(old_directions))
    matches_old_prefix = common_prefix > 0 and np.array_equal(
        directions[:common_prefix], old_directions[:common_prefix])
    validation_count = min(args.samples, len(old_directions) if matches_old_prefix else 16)
    np.save(output/'directions.npy', directions)
    references = {}
    if matches_old_prefix:
        for method in METHODS:
            with (root/f'{method}_evaluation.csv').open(encoding='utf-8', newline='') as f:
                references[method] = list(csv.DictReader(f))
            assert all(int(r['direction_index']) == i+1 for i, r in enumerate(references[method]))
    final_hashes = {method: file_hash(root/f'{method}_final.npz') for method in METHODS}
    metadata = {'status': 'running', 'samples': args.samples, 'direction_seed': args.seed,
                'case_seed': meta['seed'], 'python': sys.executable,
                'gurobi_version': list(gp.gurobi.version()), 'solver_options': EVALUATION_OPTIONS,
                'final_polytope_sha256': final_hashes,
                'directions_sha256': hashlib.sha256(directions.tobytes()).hexdigest(),
                'validation_reference': ('saved_original_interface_prefix' if matches_old_prefix
                                         else 'fresh_original_interface_first16'),
                'validation_directions': validation_count,
                'source_sha256': file_hash(Path(__file__)),
                'feasibility_count_threshold': FEASIBILITY_COUNT_THRESHOLD,
                'definition': 'Real squared-distance errors of fixed final polytopes; no gradients or training.',
                'implementation': 'Import unchanged Pyomo models via gurobi_persistent; '
                                  'native setObjective/reset(0)/optimize with explicit identical tolerances. '
                                  'One common original-domain support LP per direction for both methods.',
                'timing_note': 'Combined evaluation wall time, not training or a method speed benchmark.'}
    write_evaluation_json(output/'metadata.json', metadata)
    agg = Aggregator(seed=meta['seed'], discrete_rate=0.)
    agg.gen_CL(5)
    agg.gen_TCL(40)
    agg.gen_PV(5)
    source = agg.case_aggregator(build_only=True)
    original = NativeDomain(source)
    domains = {}
    for method in METHODS:
        with np.load(root/f'{method}_final.npz', allow_pickle=False) as data:
            calculator = ErrorCalculator({'model': source}, data['A'].copy(), data['b'].copy())
        if not matches_old_prefix:
            references[method] = reference_errors(calculator, directions[:validation_count])
            write_evaluation_json(output/f'{method}_reference_prefix.json', references[method])
        domains[method] = NativeDomain(calculator.approx_model)
    print(f'Output: {output}', flush=True)
    print(f'Original model: {original.model.NumVars} variables, '
          f'{original.model.NumConstrs} constraints', flush=True)
    rows = []
    validation = None
    started = time.perf_counter()
    with (output/'errors.csv').open('w', encoding='utf-8', newline='') as f:
        writer = None
        for index, direction in enumerate(directions):
            row = {'direction_index': index+1}
            org_support, support_violation = original.support(direction)
            for method in METHODS:
                domain = domains[method]
                apx_support, v1 = domain.support(direction)
                org_projection, v2 = original.project(apx_support)
                apx_projection, v3 = domain.project(org_support)
                row[f'{method}_feasibility_error'] = float(np.sum((apx_support-org_projection)**2))
                row[f'{method}_optimality_error'] = float(np.sum((org_support-apx_projection)**2))
                row[f'{method}_max_solver_constraint_violation'] = max(support_violation, v1, v2, v3)
            rows.append(row)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            if index+1 == validation_count:
                validation = validate_prefix(rows, references)
                write_evaluation_json(output/'prefix_validation.json', validation)
                print(f'Prefix validation passed: {len(rows)} shared directions', flush=True)
            if (index+1) % 250 == 0 or index+1 == args.samples:
                f.flush()
                elapsed = time.perf_counter()-started
                eta = elapsed * (args.samples/(index+1)-1)
                counts = {m: sum(r[f'{m}_feasibility_error'] > FEASIBILITY_COUNT_THRESHOLD
                                 for r in rows) for m in METHODS}
                print(f'{index+1}/{args.samples}: elapsed={elapsed:.1f}s, ETA={eta:.1f}s, '
                      f'feasibility_error>1e-4 counts={counts}', flush=True)
    elapsed = time.perf_counter()-started
    current_hashes = {method: file_hash(root/f'{method}_final.npz') for method in METHODS}
    if current_hashes != final_hashes:
        raise RuntimeError('Final polytopes changed during evaluation')
    result = {'metadata': metadata, 'prefix_validation': validation,
              'combined_evaluation_seconds': elapsed, 'evaluation': {}}
    for method in METHODS:
        result['evaluation'][method] = {
            f'{kind}_error': statistics([r[f'{method}_{kind}_error'] for r in rows],
                                       feasibility=kind == 'feasibility')
            for kind in ('feasibility', 'optimality')}
    exact_opt = np.array([r['exact_optimality_error'] for r in rows])
    approx_opt = np.array([r['approximate_optimality_error'] for r in rows])
    result['paired_optimality'] = {
        'exact_lower_count': int(np.count_nonzero(exact_opt < approx_opt)),
        'mean_exact_minus_approximate': float(np.mean(exact_opt-approx_opt)),
        'relative_mean_reduction_percent': float((1-exact_opt.mean()/approx_opt.mean())*100)
        if approx_opt.mean() > 0 else None}
    result['solver_quality'] = {
        'original': {'solves': original.solve_count, 'maximum_violation': original.maximum_violation},
        **{m: {'solves': domain.solve_count, 'maximum_violation': domain.maximum_violation}
           for m, domain in domains.items()}}
    metadata['status'] = 'completed'
    metadata['final_polytopes_unchanged'] = True
    metadata['same_directions_both_methods'] = True
    write_evaluation_json(output/'metadata.json', metadata)
    write_evaluation_json(output/'summary.json', result)
    print(json.dumps(result['evaluation'], ensure_ascii=False, indent=2), flush=True)
    print(f'Completed: {output}; combined evaluation {elapsed:.3f}s', flush=True)
    return output



def configure_evaluation(calculator):
    calculator.configure(feas_tol=0., opt_tol=0., active_tol=ACTIVE_TOL)
    for solver in (calculator.solver, calculator.cvx_solver):
        solver.options.update(EVALUATION_OPTIONS)
    return calculator


def evaluate(calculator, A, b, direction, kind):
    """Return the true squared error and the approximate-point active rows."""
    calculator.update_polytope(A_hat=A, b_hat=b)
    if kind == 'feasibility':
        x_apx = calculator.optimize_direction(direction, in_approx=True)
        x_org = calculator.project(x_apx) if x_apx is not None else None
    else:
        x_org = calculator.optimize_direction(direction)
        x_apx = calculator.project(x_org, to_approx=True) if x_org is not None else None
    if x_apx is None or x_org is None:
        raise RuntimeError(f'{kind}: finite-difference LP/QP solve failed')
    error = float(np.sum((x_apx - x_org) ** 2))
    if not np.isfinite(error):
        raise RuntimeError(f'{kind}: non-finite true error')
    return {'error': error,
            'active_indices': calculator._find_active(x_apx).tolist()}


def normalized(vector):
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm == 0.:
        raise RuntimeError('Cannot normalize a zero or non-finite check direction')
    return vector / norm


def check_gradients(args):
    if any(not np.isfinite(h) or h <= 0. for h in args.steps):
        raise ValueError('All --steps values must be finite and positive')
    result_dir = args.result_dir.resolve()
    paths = sorted(result_dir.glob('same_point_*.npz'))
    if not paths:
        raise ValueError(f'No same_point_*.npz snapshots found in {result_dir}')
    metadata_path = result_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    if metadata.get('status', 'completed') != 'completed':
        raise ValueError('The paired benchmark must finish before this solver diagnostic runs')
    seed = int(metadata.get('seed', 0))
    agg = Aggregator(seed=seed, discrete_rate=0.)
    agg.gen_CL(5)
    agg.gen_TCL(40)
    agg.gen_PV(5)
    original = {'model': agg.case_aggregator(build_only=True)}

    rows = []
    centers = []
    for path in paths:
        iteration = int(path.stem.rsplit('_', 1)[1])
        with np.load(path, allow_pickle=False) as snapshot:
            A = snapshot['A'].copy()
            b = snapshot['b'].copy()
            direction = snapshot['direction'].copy()
            if A.shape != (94, 24) or b.shape != (94,) or direction.shape != (24,):
                raise ValueError(f'Unexpected A/b/v shapes in {path.name}')
            calculator = configure_evaluation(ErrorCalculator(original, A.copy(), b.copy()))
            for kind_index, kind in enumerate(('feasibility', 'optimality')):
                saved_error = float(np.sum((snapshot[f'{kind}_x_apx'] -
                                            snapshot[f'{kind}_x_org']) ** 2))
                stored_active = snapshot[f'{kind}_active_indices'].tolist()
                center = evaluate(calculator, A, b, direction, kind)
                base = {'snapshot': path.name, 'iteration': iteration, 'kind': kind,
                        'saved_error': saved_error, 'center_error': center['error'],
                        'center_error_abs_difference': abs(center['error'] - saved_error),
                        'center_error_matches_saved': bool(np.isclose(
                            center['error'], saved_error, rtol=1e-7, atol=1e-10)),
                        'stored_active_indices': stored_active,
                        'center_active_indices': center['active_indices'],
                        'center_active_matches_saved': center['active_indices'] == stored_active}
                centers.append(base.copy())
                if saved_error <= ERROR_THRESHOLD or center['error'] <= ERROR_THRESHOLD:
                    rows.append({**base, 'status': 'skipped_small_error',
                                 'note': 'Saved or re-solved squared error <= 1e-8; '
                                         'solver roundoff can dominate this derivative check.'})
                    print(f'{path.name} {kind}: skipped (e={center["error"]:.3e})', flush=True)
                    continue

                exact = np.concatenate((snapshot[f'{kind}_gradient_A'].ravel(),
                                        snapshot[f'{kind}_gradient_b'].ravel()))
                approximate = np.concatenate((snapshot[f'{kind}_surrogate_gradient_A'].ravel(),
                                              snapshot[f'{kind}_surrogate_gradient_b'].ravel()))
                if not np.all(np.isfinite(exact)) or not np.all(np.isfinite(approximate)):
                    raise ValueError(f'Non-finite saved gradient in {path.name} ({kind})')
                random_seed = seed + 20000 + 2 * iteration + kind_index
                rng = np.random.RandomState(random_seed)
                check_directions = (
                    ('normalized_exact_gradient', normalized(exact)),
                    ('normalized_random_joint_A_b', normalized(rng.randn(exact.size))),
                )
                for check_name, delta in check_directions:
                    dA = delta[:A.size].reshape(A.shape)
                    db = delta[A.size:]
                    exact_dot = float(exact @ delta)
                    approximate_dot = float(approximate @ delta)
                    for h in args.steps:
                        plus = evaluate(calculator, A + h * dA, b + h * db, direction, kind)
                        minus = evaluate(calculator, A - h * dA, b - h * db, direction, kind)
                        fd = (plus['error'] - minus['error']) / (2. * h)
                        scale = max(abs(fd), abs(exact_dot), 1e-12)
                        active_stable = (plus['active_indices'] == center['active_indices'] ==
                                         minus['active_indices'])
                        row = {**base, 'status': 'checked', 'check_direction': check_name,
                               'random_direction_seed': random_seed if 'random' in check_name else None,
                               'h': h, 'direction_joint_l2_norm': float(np.linalg.norm(delta)),
                               'plus_error': plus['error'], 'minus_error': minus['error'],
                               'finite_difference': fd, 'exact_dot': exact_dot,
                               'approximate_dot': approximate_dot,
                               'exact_absolute_error': abs(fd - exact_dot),
                               'approximate_absolute_error': abs(fd - approximate_dot),
                               'relative_error_scale': scale,
                               'exact_relative_error': abs(fd - exact_dot) / scale,
                               'approximate_relative_error': abs(fd - approximate_dot) / scale,
                               'plus_active_indices': plus['active_indices'],
                               'minus_active_indices': minus['active_indices'],
                               'plus_active_matches_center': plus['active_indices'] == center['active_indices'],
                               'minus_active_matches_center': minus['active_indices'] == center['active_indices'],
                               'active_set_stable': active_stable,
                               'note': '' if active_stable else
                               'Active rows changed under the finite-difference perturbation; '
                               'do not interpret discrepancy alone as an incorrect local derivative.'}
                        rows.append(row)
                        print(f'{path.name} {kind} {check_name} h={h:g}: '
                              f'FD={fd:.6e}, exact={exact_dot:.6e}, approx={approximate_dot:.6e}, '
                              f'active_stable={active_stable}', flush=True)

    report = {'status': 'completed', 'python': sys.executable, 'seed': seed,
              'solver_options': EVALUATION_OPTIONS, 'active_tol': ACTIVE_TOL,
              'error_threshold': ERROR_THRESHOLD, 'steps': args.steps,
              'snapshots': [path.name for path in paths],
              'definition': 'Central differences of the true squared distance error e at the saved '
                            'normalized A,b. Perturbations have joint A/b Euclidean norm 1. '
                            'Saved surrogate gradients are compared to the derivative of e, not of L.',
              'relative_error_definition': '|FD - gradient_dot| / max(|FD|, |exact_dot|, 1e-12), '
                                           'with the same denominator for both methods.',
              'active_set_note': 'Stability compares approximate LP-vertex active rows for feasibility '
                                 'and approximate projection active rows for optimality, using active_tol. '
                                 'A changed set can indicate nonsmoothness or active-tolerance effects; '
                                 'an unchanged set alone is not a complete differentiability certificate.',
              'checked_count': sum(row['status'] == 'checked' for row in rows),
              'skipped_branch_count': sum(row['status'] != 'checked' for row in rows),
              'centers': centers, 'checks': rows}
    json_path = result_dir / 'finite_difference_checks.json'
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                         encoding='utf-8')
    columns = list(dict.fromkeys(key for row in rows for key in row))
    csv_path = result_dir / 'finite_difference_checks.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, list) else value
                             for key, value in row.items()})
    print(f'Saved {json_path}\nSaved {csv_path}', flush=True)


def recheck_extremes(args):
    import gurobipy as gp
    evaluation_dir = args.evaluation_dir.resolve()
    root = evaluation_dir.parent
    output = evaluation_dir / 'recheck_extremes.json'
    if output.exists():
        raise FileExistsError(output)
    summary = json.loads((evaluation_dir / 'summary.json').read_text(encoding='utf-8'))
    metadata = summary['metadata']
    training_meta = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
    if metadata['status'] != 'completed' or training_meta['status'] != 'completed':
        raise RuntimeError('Training and the large evaluation must both be completed')
    if metadata['case_seed'] != training_meta['seed']:
        raise RuntimeError('Case seeds differ')
    if metadata['solver_options'] != training_meta['solver_options']:
        raise RuntimeError('Solver options differ from the paired training benchmark')
    if list(gp.gurobi.version()) != metadata['gurobi_version']:
        raise RuntimeError('Gurobi version differs')

    expected_hashes = metadata['final_polytope_sha256']
    initial_hashes = {method: file_hash(root / f'{method}_final.npz') for method in METHODS}
    if initial_hashes != expected_hashes:
        raise RuntimeError('A saved final polytope changed')
    directions = np.load(evaluation_dir / 'directions.npy', allow_pickle=False)
    direction_hash = hashlib.sha256(directions.tobytes()).hexdigest()
    if direction_hash != metadata['directions_sha256']:
        raise RuntimeError('Direction array hash mismatch')
    if directions.shape != (metadata['samples'], 24) or not np.all(np.isfinite(directions)):
        raise RuntimeError('Unexpected direction array')
    seeded_directions = np.random.RandomState(metadata['direction_seed']).randn(metadata['samples'], 24)
    if not np.array_equal(directions, seeded_directions):
        raise RuntimeError('Direction array does not match its recorded random seed')
    old_directions = np.load(root / 'evaluation_directions.npy', allow_pickle=False)
    common_prefix = min(len(directions), len(old_directions))
    matches_original_prefix = common_prefix > 0 and np.array_equal(
        directions[:common_prefix], old_directions[:common_prefix])
    original_seed = training_meta['evaluation_seed']
    if metadata['direction_seed'] == original_seed and not matches_original_prefix:
        raise RuntimeError('Original evaluation directions are not an identical prefix')

    with (evaluation_dir / 'errors.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(directions):
        raise RuntimeError('Error-row count differs from the direction count')
    if any(int(row['direction_index']) != i+1 for i, row in enumerate(rows)):
        raise RuntimeError('Error rows are not indexed consecutively from 1')

    selected = {}
    for method in METHODS:
        for kind in KINDS:
            column = f'{method}_{kind}_error'
            values = np.asarray([float(row[column]) for row in rows])
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise RuntimeError(f'Invalid values in {column}')
            maximum_index = int(np.argmax(values)) + 1
            expected = summary['evaluation'][method][f'{kind}_error']
            if maximum_index != expected['max_direction_index_one_based']:
                raise RuntimeError(f'Extremum index differs for {column}')
            np.testing.assert_allclose(values[maximum_index-1], expected['max'], rtol=0, atol=0)
            selected.setdefault(maximum_index, []).append(column)

    aggregator = Aggregator(seed=metadata['case_seed'], discrete_rate=0.)
    aggregator.gen_CL(5)
    aggregator.gen_TCL(40)
    aggregator.gen_PV(5)
    original = {'model': aggregator.case_aggregator(build_only=True)}
    calculators = {}
    for method in METHODS:
        with np.load(root / f'{method}_final.npz', allow_pickle=False) as data:
            calculator = ErrorCalculator(original, data['A'].copy(), data['b'].copy())
        calculator.configure(feas_tol=0., opt_tol=0., active_tol=training_meta['active_tol'])
        for solver in (calculator.solver, calculator.cvx_solver):
            solver.options.update(metadata['solver_options'])
        calculators[method] = calculator

    checked = []
    start = time.perf_counter()
    for index in sorted(selected):
        direction = directions[index-1].copy()
        item = {'direction_index_one_based': index,
                'selected_for': selected[index], 'direction': direction.tolist(),
                'methods': {}}
        for method, calculator in calculators.items():
            x_apx = calculator.optimize_direction(direction, in_approx=True)
            if x_apx is None:
                raise RuntimeError(f'Approximate-domain LP failed for {method}, direction {index}')
            q_org = calculator.project(x_apx)
            x_org = calculator.optimize_direction(direction)
            if q_org is None or x_org is None:
                raise RuntimeError(f'Original-domain solve failed for {method}, direction {index}')
            q_apx = calculator.project(x_org, to_approx=True)
            if q_apx is None:
                raise RuntimeError(f'Approximate-domain projection failed for {method}, direction {index}')
            points = {'approximate_support': x_apx, 'original_projection': q_org,
                      'original_support': x_org, 'approximate_projection': q_apx}
            if any(not np.all(np.isfinite(point)) for point in points.values()):
                raise RuntimeError('Nonfinite optimization output')
            errors = {'feasibility': float(np.sum((x_apx-q_org)**2)),
                      'optimality': float(np.sum((x_org-q_apx)**2))}
            comparisons = {}
            for kind in KINDS:
                reference = float(rows[index-1][f'{method}_{kind}_error'])
                value = errors[kind]
                np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-9,
                                           err_msg=f'{method}/{kind}, direction {index}')
                comparisons[f'{kind}_error'] = {
                    'original_interface': value, 'persistent_interface': reference,
                    'absolute_difference': abs(value-reference),
                    'relative_difference': abs(value-reference)/abs(reference) if abs(reference) > 1e-8 else None,
                    'passed': True}
            item['methods'][method] = {**comparisons,
                                       'points': {name: point.tolist() for name, point in points.items()}}
        checked.append(item)
        print(f'Rechecked direction {index}: {selected[index]}', flush=True)

    final_hashes = {method: file_hash(root / f'{method}_final.npz') for method in METHODS}
    if final_hashes != initial_hashes:
        raise RuntimeError('A final polytope changed during the independent recheck')
    result = {'status': 'passed', 'python': sys.executable,
              'gurobi_version': list(gp.gurobi.version()),
              'implementation': 'Original ErrorCalculator using SolverFactory(gurobi), fresh per-call LP/QP translation; no training.',
              'solver_options': metadata['solver_options'],
              'case_seed': metadata['case_seed'], 'direction_seed': metadata['direction_seed'],
              'tolerance': {'rtol': 1e-6, 'atol': 1e-9},
              'checked_unique_directions': len(checked),
              'all_direction_indices_verified': True,
              'all_directions_match_recorded_seed': True,
              'original_evaluation_seed': original_seed,
              'matches_original_evaluation_prefix': matches_original_prefix,
              'directions_sha256': direction_hash,
              'final_polytope_sha256': final_hashes,
              'final_polytopes_unchanged': True,
              'source_sha256': file_hash(Path(__file__)),
              'elapsed_seconds': time.perf_counter()-start,
              'rechecks': checked}
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Passed independent rechecks: {output}', flush=True)


def summarize_runs(result_dirs, output, *, samples=10000, seed=10000, axis='training'):
    """Recompute comparable metrics from saved rows; list every incomplete run."""
    if axis not in ('training', 'evaluation') or samples < 1:
        raise ValueError('Use training/evaluation axis and a positive sample count')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite summary: {output}')
    rows, excluded = [], []
    signature = None
    expected_initial = None
    expected_directions = None
    expected_final = None
    seen_seeds = set()
    for supplied in result_dirs:
        supplied = Path(supplied).resolve()
        evaluation_dir = supplied if axis == 'evaluation' else supplied / f'evaluation_{samples}_seed{seed}'
        root = evaluation_dir.parent
        try:
            train = json.loads((root / 'metadata.json').read_text(encoding='utf-8'))
            meta = json.loads((evaluation_dir / 'metadata.json').read_text(encoding='utf-8'))
            report = json.loads((evaluation_dir / 'summary.json').read_text(encoding='utf-8'))
            train_report = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
            recheck = json.loads((evaluation_dir / 'recheck_extremes.json').read_text(encoding='utf-8'))
            if train['status'] != 'completed' or meta['status'] != 'completed':
                raise ValueError('Training or evaluation is incomplete')
            if recheck['status'] != 'passed' or not report['prefix_validation']['passed']:
                raise ValueError('Independent evaluation checks have not passed')
            if not train.get('directions_identical_verified'):
                raise ValueError('Paired training-direction verification is missing')
            if meta['samples'] != samples:
                raise ValueError('Evaluation sample count differs from the requested protocol')
            if axis == 'training' and meta['direction_seed'] != seed:
                raise ValueError('Evaluation seed differs from the requested protocol')
            if (meta['case_seed'] != train['seed'] or
                    meta['solver_options'] != train['solver_options'] or
                    recheck['solver_options'] != meta['solver_options']):
                raise ValueError('Evaluation or recheck uses a different case or solver settings')
            final = {method: file_hash(root / f'{method}_final.npz') for method in METHODS}
            if final != meta['final_polytope_sha256'] or final != recheck['final_polytope_sha256']:
                raise ValueError('Final polytope hash mismatch')
            directions = np.load(evaluation_dir / 'directions.npy', allow_pickle=False)
            direction_hash = hashlib.sha256(directions.tobytes()).hexdigest()
            if (direction_hash != meta['directions_sha256'] or
                    direction_hash != recheck['directions_sha256']):
                raise ValueError('Evaluation-direction hash mismatch')
            expected = np.random.RandomState(meta['direction_seed']).randn(samples, 24)
            if not np.array_equal(directions, expected):
                raise ValueError('Evaluation directions differ from their recorded seed')
            planned = np.load(root / 'training_directions.npy', allow_pickle=False)
            if planned.shape != (train['iterations'], 24):
                raise ValueError('Training-direction count does not match actual update count')
            training_seed = train.get('training_seed', train['seed'])
            if not np.array_equal(planned, np.random.RandomState(training_seed).randn(train['iterations'], 24)):
                raise ValueError('Training directions differ from their recorded seed')
            if hashlib.sha256(planned.tobytes()).hexdigest() != train['training_directions_sha256']:
                raise ValueError('Training-direction hash mismatch')
            for method in METHODS:
                used = np.load(root / f'{method}_used_directions.npy', allow_pickle=False)
                if not np.array_equal(used, planned):
                    raise ValueError(f'{method} did not use the common training directions')
            current_signature = json.dumps({
                k: train.get(k) for k in ('case_seed', 'initialization_seed', 'iterations',
                    'stages', 'solver_options', 'active_tol', 'error_cutoff_both_methods',
                    'dtype', 'model', 'device', 'device_name', 'gurobi', 'torch', 'numpy',
                    'polytope_shape', 'active_basis_policy', 'initialization_note',
                    'iteration_order', 'source_sha256', 'exact_gradient_configuration')}, sort_keys=True)
            with np.load(root / 'initial_polytope.npz', allow_pickle=False) as initial:
                initial_hash = hashlib.sha256(initial['A'].tobytes() + initial['b'].tobytes()).hexdigest()
            with (evaluation_dir / 'errors.csv').open(encoding='utf-8', newline='') as stream:
                errors = list(csv.DictReader(stream))
            if len(errors) != samples or any(int(r['direction_index']) != i + 1 for i, r in enumerate(errors)):
                raise ValueError('Incomplete or incorrectly indexed evaluation rows')
            row = {'training_seed': training_seed,
                   'evaluation_seed': meta['direction_seed'], 'samples': samples,
                   'run': os.path.relpath(root, output), 'evaluation': os.path.relpath(evaluation_dir, output)}
            for method in METHODS:
                for kind in KINDS:
                    values = np.asarray([float(r[f'{method}_{kind}_error']) for r in errors])
                    if not np.all(np.isfinite(values)) or np.any(values < 0):
                        raise ValueError('Nonfinite or negative saved errors')
                    saved = report['evaluation'][method][f'{kind}_error']
                    np.testing.assert_allclose([values.mean(), values.max()], [saved['mean'], saved['max']], rtol=1e-12, atol=1e-15)
                    row[f'{method}_mean_{kind}'] = float(values.mean())
                    row[f'{method}_max_{kind}'] = float(values.max())
                    if kind == 'feasibility':
                        row[f'{method}_feasibility_count_gt_1e_4'] = int(np.count_nonzero(values > FEASIBILITY_COUNT_THRESHOLD))
                        if row[f'{method}_feasibility_count_gt_1e_4'] != saved['count_gt_1e-4']:
                            raise ValueError('Saved >1e-4 count differs from the raw data')
                row[f'{method}_training_seconds'] = train_report['training'][method]['step_seconds']
            identity = row['training_seed'] if axis == 'training' else row['evaluation_seed']
            if identity in seen_seeds:
                raise ValueError(f'Duplicate {axis} seed {identity}')
            if signature is not None and (signature != current_signature or expected_initial != initial_hash):
                raise ValueError('Training protocol or initial polytope differs; summarize separately')
            if axis == 'training' and expected_directions not in (None, direction_hash):
                raise ValueError('Training-seed comparison requires the same evaluation directions')
            if axis == 'evaluation' and expected_final not in (None, final):
                raise ValueError('Evaluation-seed comparison requires the same final polytopes')
            signature, expected_initial = current_signature, initial_hash
            expected_directions, expected_final = direction_hash, final
            seen_seeds.add(identity)
            rows.append(row)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            excluded.append({'path': str(supplied), 'status': 'incomplete_or_incompatible', 'reason': str(exc)})
    metrics = ('mean_feasibility', 'max_feasibility', 'feasibility_count_gt_1e_4',
               'mean_optimality', 'max_optimality', 'training_seconds')
    comparison = {}
    for metric in metrics:
        a = np.asarray([r[f'approximate_{metric}'] for r in rows], dtype=float)
        e = np.asarray([r[f'exact_{metric}'] for r in rows], dtype=float)
        comparison[metric] = {
            'approximate_mean': float(a.mean()) if len(a) else None,
            'exact_mean': float(e.mean()) if len(e) else None,
            'approximate_wins': int(np.count_nonzero(a < e)),
            'exact_wins': int(np.count_nonzero(e < a)),
            'ties': int(np.count_nonzero(a == e)),
        }
    result = {'axis': axis, 'included_count': len(rows), 'excluded_count': len(excluded),
              'feasibility_count_threshold': FEASIBILITY_COUNT_THRESHOLD,
              'averaging': 'Equal weight per complete seed; exclusions are listed, never replaced.',
              'comparison': comparison, 'runs': rows, 'excluded': excluded}
    output.mkdir(parents=True, exist_ok=False)
    write_evaluation_json(output / 'summary.json', result)
    if rows:
        with (output / 'comparison.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [f'{axis} seeds: {len(rows)} complete; {len(excluded)} excluded.',
             'Feasibility count: squared error strictly > 1e-4.']
    for metric, values in comparison.items():
        lines.append(f'{metric}: {json.dumps(values, ensure_ascii=False)}')
    for item in excluded:
        lines.append(f'EXCLUDED {item["path"]}: {item["reason"]}')
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return result


def audit_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    evaluation = commands.add_parser('evaluate', help='Evaluate fixed final polytopes')
    evaluation.add_argument('result_dir', type=Path)
    evaluation.add_argument('--samples', type=int, default=10000)
    evaluation.add_argument('--seed', type=int, default=10000)
    evaluation.add_argument('--recheck', action='store_true', help='Independently re-solve extreme directions')
    check = commands.add_parser('check', help='Finite-difference audit of saved gradient snapshots')
    check.add_argument('result_dir', type=Path)
    check.add_argument('--steps', type=float, nargs='+', default=[1e-5, 1e-6])
    recheck = commands.add_parser('recheck', help='Independently check completed evaluation extrema')
    recheck.add_argument('evaluation_dir', type=Path)
    summary = commands.add_parser('summarize', help='Summarize complete runs or evaluation-seed directories')
    summary.add_argument('result_dirs', type=Path, nargs='+')
    summary.add_argument('--axis', choices=('training', 'evaluation'), default='training')
    summary.add_argument('--samples', type=int, default=10000)
    summary.add_argument('--seed', type=int, default=10000)
    summary.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'evaluate':
        output = evaluate_final(args)
        if args.recheck:
            recheck_extremes(argparse.Namespace(evaluation_dir=output))
    elif args.command == 'check':
        check_gradients(args)
    elif args.command == 'recheck':
        recheck_extremes(args)
    else:
        result = summarize_runs(args.result_dirs, args.output, samples=args.samples, seed=args.seed, axis=args.axis)
        return 1 if result['excluded_count'] or not result['included_count'] else 0
    return 0



def main(argv=None):
    """Run one part of the paired-gradient case."""
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = ("train", "evaluate", "check", "recheck", "summarize")
    if not argv or argv[0] in ("-h", "--help"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("command", choices=commands, help="Use COMMAND --help for its original options")
        parser.parse_args(argv)
        return 0
    if argv[0] == "train":
        return train_main(argv[1:])
    if argv[0] in commands[1:]:
        return audit_main(argv)
    if argv[0].startswith("-"):
        return train_main(argv)
    raise SystemExit("Unknown gradient-case command: " + argv[0])


if __name__ == "__main__":
    raise SystemExit(main())
