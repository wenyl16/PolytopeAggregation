"""Ablate MT dwell time on precisely the previous 5,000 physical targets."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('code').resolve()))
from Simulator.testers.evaluate_microgrid_nonconvexity import (
    load_case, sample_row, require_witness, write_json, write_npz, digest)
from Simulator.testers._nonconvexity_domain import PhysicalDomain
from Simulator.testers._nonconvexity_sampling import build_summary
from Simulator.testers.verify_microgrid_nonconvexity import physical_validator

base = Path('code/results/microgrid/scale3')
old = base / 'nonconvexity'
out = base / 'nonconvexity_no_min_uptime'
mg = load_case(base / 'training', device_scale=3, disable_mt_min_uptime=True)
def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
endpoint, probes, saved = [arrays(old / name) for name in
                          ('endpoints.npz', 'probes.npz', 'projections.npz')]
old_rows = pd.read_csv(old / 'samples.csv')
checkpoint = json.loads((old / 'checkpoint.json').read_text(encoding='utf-8'))
assert checkpoint['completed'] == 5000
for name, sha in checkpoint['files'].items():
    assert digest(old / name) == sha
inputs = {name: digest(old / name) for name in
          ('metadata.json', 'endpoints.npz', 'probes.npz', 'projections.npz',
           'samples.csv', 'independent_storage_verification.json')}
rows, diagnostics, points, primals = [], [], [], []
started = time.perf_counter()
with PhysicalDomain(mg.model) as domain:
    assert list(domain.names) == endpoint['variable_names'].tolist()
    np.testing.assert_array_equal(endpoint['p_min'], mg.pmin)
    np.testing.assert_array_equal(endpoint['p_max'], mg.pmax)
    validate = physical_validator(mg.model, domain.names)
    assert validate(endpoint['raw_primal'])['passed']
    assert validate(saved['primal'])['passed']
    for i in range(5000):
        answer = domain.project(probes['targets'][i], start=saved['primal'][i])
        require_witness(domain, answer, f'paired probe {i}')
        row = sample_row(i, str(probes['kind'][i]), answer,
                         bool(old_rows.iloc[i]['same_binary_mode']))
        assert row['upper_squared'] <= old_rows.iloc[i]['upper_squared'] + 1e-10
        row['classification'] = ('below_threshold' if row['upper_percent'] <= .01 else
                                 'detected_gap' if row['lower_percent'] > .01 else 'undetermined')
        rows.append(row)
        diagnostics.append(answer['diagnostics'])
        points.append(answer['point'])
        primals.append(answer['primal'])
        if (i + 1) % 500 == 0:
            print(json.dumps({'paired_completed': i + 1,
                              'max_upper_percent': max(r['upper_percent'] for r in rows)}), flush=True)
    primals, points = np.asarray(primals), np.asarray(points)
    verification = validate(primals)
    assert verification['passed']
    ledger_error = 0.
    for j, ess in enumerate(mg.ESS_list):
        names = list(domain.names)
        ch = primals[:, [names.index(f'p_chg_ESS[{j},{t}]') for t in range(24)]]
        dis = primals[:, [names.index(f'p_dis_ESS[{j},{t}]') for t in range(24)]]
        soc = primals[:, [names.index(f'e_ESS[{j},{t}]') for t in range(25)]]
        expected = np.c_[np.full(5000, ess['init_SOC']), ess['init_SOC'] +
                         np.cumsum(mg.deltaT * (ess['eta_chg'] * ch - dis / ess['eta_dis']) /
                                   ess['E_rated'], axis=1)]
        ledger_error = max(ledger_error, float(np.max(np.abs(soc - expected))))
    assert ledger_error < 1e-8
summary = build_summary(rows, threshold_percent=.01)
summary.update(completed=5000, scope='Exactly the baseline targets and normalization; only MT minimum uptime removed',
               original_inputs_sha256=inputs, physical_validation=verification,
               max_cumulative_ESS_SOC_residual=ledger_error,
               all_360_original_binary_variables_free=True,
               removed_constraints=mg.nonconvexity_removed_constraints,
               wall_seconds=time.perf_counter() - started,
               runner_sha256=digest(Path(__file__)))
assert inputs == {name: digest(old / name) for name in inputs}
pd.DataFrame(rows).to_csv(out / 'paired_samples.csv', index=False)
write_npz(out / 'paired_projections.npz', points=points, primal=primals)
write_json(out / 'paired_projection_diagnostics.json', diagnostics)
write_json(out / 'paired_summary.json', summary)
print(json.dumps(summary['groups']['overall']), flush=True)
