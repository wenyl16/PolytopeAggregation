"""Solver-free equation regression tests for microgrid ESS and MT constraints.

Preserves the complete ESS interval-energy and MT minimum-uptime test cases.
Actual Pyomo constraint expressions are checked against independent cumulative
energy and forward-run oracles. No optimization calls are made.

Run: python -m unittest Simulator.cases.test_microgrid_constraints -v
"""

from itertools import product
from pathlib import Path
import sys
import unittest

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pyomo.environ as pyo

from Simulator.cases.microgrid_case import MicroGrid
from Simulator.cases.microgrid_count_solver import CountBranchingSolver
from Simulator.testers.case_microgrid_nonconvexity import expected_model_size

HORIZON = 24


def make_storage_case(T=24):
    data = {name + "_list": [] for name in ("MT", "PV", "TL", "CL", "ESS", "HP")}
    data["ESS_list"] = [{"Pchg": 100., "Pdis": 100., "init_SOC": .5,
                         "eta_chg": .95, "eta_dis": .98, "E_rated": 250.,
                         "SOC_min": .1, "SOC_max": 1., "baseline": np.zeros(T),
                         "cost_chg": .05, "cost_dis": .05}]
    mg = MicroGrid(seed=42, T=T, data=data)
    mg.build_mg_constraints(conservativeness=0.)
    return mg


def assign_energy_ledger(mg, charge, discharge):
    charge, discharge = np.asarray(charge), np.asarray(discharge)
    states = np.r_[.5, .5 + np.cumsum((.95 * charge - discharge / .98) * mg.deltaT) / 250.]
    for boundary, state in enumerate(states):
        mg.model.e_ESS[0, boundary].set_value(float(state))
    for t in range(mg.T):
        mg.model.p_chg_ESS[0, t].set_value(float(charge[t]))
        mg.model.p_dis_ESS[0, t].set_value(float(discharge[t]))
        mg.model.u_ESS[0, t].set_value(int(charge[t] > 0))
        mg.model.var_proj[t].set_value(float((discharge[t] - charge[t] - mg.pmin[t]) /
                                           (mg.pmax[t] - mg.pmin[t])))
    return states


def violations(model):
    bad = []
    for row in model.component_data_objects(pyo.Constraint, active=True):
        value = float(pyo.value(row.body))
        if ((row.has_lb() and value < float(pyo.value(row.lower)) - 1e-9) or
                (row.has_ub() and value > float(pyo.value(row.upper)) + 1e-9)):
            bad.append(row.name)
    return bad


class StorageTimeIndexTests(unittest.TestCase):
    def test_initial_boundary_and_all24_intervals_exist(self):
        mg = make_storage_case()
        self.assertEqual(list(mg.model.T), list(range(24)))
        self.assertEqual(list(mg.model.ESS_state_times), list(range(25)))
        self.assertEqual(list(mg.model.ess_energy_dynamics_0), list(range(24)))
        counts = {name: int(name == "ESS") for name in ("MT", "PV", "TL", "CL", "ESS", "HP")}
        variables = list(mg.model.component_data_objects(pyo.Var))
        self.assertEqual((len(variables), sum(v.is_binary() for v in variables)),
                         expected_model_size(counts))

    def test_first_interval_discharge_is_deducted_after_initial_state(self):
        mg = make_storage_case()
        charge, discharge = np.zeros(24), np.zeros(24)
        discharge[0] = 98.
        charge[1:3] = (100., 100. / .95 - 100.)
        states = assign_energy_ledger(mg, charge, discharge)
        self.assertAlmostEqual(states[0], .5)
        self.assertAlmostEqual(states[1], .1)
        self.assertAlmostEqual(states[-1], .5)
        self.assertEqual(violations(mg.model), [])

    def test_first_interval100kW_violates_available_energy(self):
        mg = make_storage_case()
        charge, discharge = np.zeros(24), np.zeros(24)
        discharge[0] = 100.
        charge[1:3] = (100., 100. / (.98 * .95) - 100.)
        assign_energy_ledger(mg, charge, discharge)
        self.assertIn("ess_soc_limit_0[1]", violations(mg.model))

    def test_first_interval_power_cannot_leave_all_states_unchanged(self):
        mg = make_storage_case()
        charge, discharge = np.zeros(24), np.zeros(24)
        discharge[0] = 50.
        assign_energy_ledger(mg, charge, discharge)
        for state in mg.model.e_ESS.values():
            state.set_value(.5)
        self.assertIn("ess_energy_dynamics_0[0]", violations(mg.model))

    def test_first_charge_changes_E1_but_not_E0(self):
        mg = make_storage_case()
        charge, discharge = np.zeros(24), np.zeros(24)
        charge[0] = 10.
        states = assign_energy_ledger(mg, charge, discharge)
        self.assertAlmostEqual(states[0], .5)
        self.assertAlmostEqual(states[1], .538)
        self.assertEqual(violations(mg.model), [])

    def test_last_interval_updates_terminal_state_and_terminal_requirement(self):
        mg = make_storage_case()
        charge, discharge = np.zeros(24), np.zeros(24)
        discharge[-1] = 1.
        states = assign_energy_ledger(mg, charge, discharge)
        self.assertAlmostEqual(states[-2], .5)
        self.assertLess(states[-1], .5)
        self.assertEqual(violations(mg.model), ["ess_final_soc_0"])
        mg.model.e_ESS[0, 24].set_value(.5)
        self.assertIn("ess_energy_dynamics_0[23]", violations(mg.model))

    def test_quarter_hour_energy_and_efficiencies(self):
        mg = make_storage_case(96)
        charge, discharge = np.zeros(96), np.zeros(96)
        discharge[0] = 10.
        charge[1] = 10. / (.98 * .95)
        states = assign_energy_ledger(mg, charge, discharge)
        self.assertAlmostEqual(states[1], .5 - 10. * .25 / (.98 * 250.))
        self.assertAlmostEqual(states[-1], .5)
        self.assertEqual(violations(mg.model), [])

    def test_redundant_headroom_uses_initial_and_last_boundary(self):
        mg = make_storage_case()
        backend = CountBranchingSolver(storage_parameters=mg.ESS_list, deltaT=mg.deltaT)
        specs = backend._storage_specs(mg.model)
        self.assertEqual(len(specs), 24)
        self.assertIs(specs[0][2], mg.model.e_ESS[0, 0])
        self.assertIs(specs[-1][2], mg.model.e_ESS[0, 23])
        self.assertFalse(backend.diagnostics["storage_headroom_t0_skipped"])


def make_mt_model(min_on_time, initial_status):
    """Keep the real one-hour model and its real MT constraint block."""
    data = {name + "_list": [] for name in ("MT", "PV", "TL", "CL", "ESS", "HP")}
    data["MT_list"] = [{"P_min": 20.0, "P_max": 100.0,
                        "a": 0.015, "b": 0.15, "c": 5.0,
                        "min_on_time": min_on_time, "initial_status": initial_status,
                        "baseline": np.zeros(HORIZON)}]
    mg = MicroGrid(seed=42, T=HORIZON, data=data)
    mg.build_mg_constraints(conservativeness=0.0)
    return mg.model


def violated_mt_rows(model, schedule, initial_status):
    """Assign a complete binary schedule and evaluate the original MT rows."""
    if len(schedule) != HORIZON or any(value not in (0, 1) for value in schedule):
        raise ValueError("A complete 24-period binary schedule is required")
    previous = initial_status
    for t, online in enumerate(schedule):
        model.u_MT[0, t].set_value(online)
        model.y_MT[0, t].set_value(int(online == 1 and previous == 0))
        model.z_MT[0, t].set_value(int(online == 0 and previous == 1))
        model.p_MT[0, t].set_value(20.0 * online)
        previous = online
    violated = []
    for row in model.component_data_objects(pyo.Constraint, active=True):
        if not row.parent_component().local_name.startswith("mt_"):
            continue
        activity = float(pyo.value(row.body))
        if ((row.has_lb() and activity < float(pyo.value(row.lower)) - 1e-9) or
                (row.has_ub() and activity > float(pyo.value(row.upper)) + 1e-9)):
            violated.append(row.name)
    return violated


def forward_run_oracle(schedule, min_on_time, initial_status):
    """Check runs looking forward from transitions, including visible endings."""
    if initial_status and not all(schedule[:min(min_on_time, len(schedule))]):
        return False
    previous = initial_status
    for start, online in enumerate(schedule):
        if online and not previous:
            end = min(start + min_on_time, len(schedule))
            if not all(schedule[start:end]):
                return False
        previous = online
    return True


class MicrogridMinimumUptimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = {(uptime, initial): make_mt_model(uptime, initial)
                      for uptime, initial in ((1, 0), (1, 1), (4, 0), (4, 1),
                                              (HORIZON + 6, 0), (HORIZON + 6, 1))}

    def assert_schedule(self, schedule, expected, uptime=4, initial=0):
        violations = violated_mt_rows(self.models[uptime, initial], schedule, initial)
        self.assertEqual(not violations, expected,
                         f"U={uptime}, initial={initial}, schedule={schedule}, violations={violations}")
        self.assertEqual(forward_run_oracle(schedule, uptime, initial), expected)

    @staticmethod
    def single_run(start, length):
        schedule = [0] * HORIZON
        schedule[start:start + length] = [1] * length
        return schedule

    def test_first_period_start_requires_four_visible_periods(self):
        for length in (1, 2, 3, 4):
            with self.subTest(length=length):
                self.assert_schedule(self.single_run(0, length), length == 4)

    def test_early_and_interior_startups_cannot_stop_early(self):
        for start in (1, 2, 7, 12):
            for length in (1, 2, 3, 4):
                with self.subTest(start=start, length=length):
                    self.assert_schedule(self.single_run(start, length), length == 4)

    def test_multiple_legal_runs_and_all_off(self):
        schedule = [0] * HORIZON
        schedule[0:4] = [1] * 4
        schedule[5:9] = [1] * 4
        self.assert_schedule(schedule, True)
        self.assert_schedule(self.single_run(0, 8), True)
        self.assert_schedule([0] * HORIZON, True)

    def test_terminal_startups_only_require_the_visible_horizon(self):
        for remaining in (1, 2, 3, 4):
            with self.subTest(visible_periods=remaining):
                self.assert_schedule(self.single_run(HORIZON - remaining, remaining), True)
        # A visible shutdown after a one-period run remains a violation.
        self.assert_schedule(self.single_run(HORIZON - 2, 1), False)

    def test_initially_online_conservative_window_includes_period_zero(self):
        for uptime in (1, 4):
            with self.subTest(uptime=uptime):
                self.assert_schedule([0] + [1] * (HORIZON - 1), False,
                                     uptime=uptime, initial=1)
                self.assert_schedule([1] * uptime + [0] * (HORIZON - uptime), True,
                                     uptime=uptime, initial=1)
        for shutdown_at in (1, 2, 3):
            with self.subTest(shutdown_at=shutdown_at):
                schedule = [1] * HORIZON
                schedule[shutdown_at] = 0
                self.assert_schedule(schedule, False, initial=1)

    def test_one_period_uptime_allows_each_visible_transition(self):
        self.assert_schedule([int(t % 2 == 0) for t in range(HORIZON)], True, uptime=1)
        self.assert_schedule([int(t % 2 == 1) for t in range(HORIZON)], True, uptime=1)

    def test_uptime_longer_than_horizon_does_not_create_out_of_range_rows(self):
        uptime = HORIZON + 6
        self.assert_schedule([1] * HORIZON, True, uptime=uptime, initial=1)
        self.assert_schedule([1] * (HORIZON - 1) + [0], False, uptime=uptime, initial=1)
        self.assert_schedule(self.single_run(2, HORIZON - 2), True, uptime=uptime)
        self.assert_schedule(self.single_run(2, HORIZON - 3), False, uptime=uptime)

    def test_all_short_prefix_schedules_match_independent_forward_oracle(self):
        # Hold the final prefix state through T=24. This preserves the true
        # one-hour device model and exactly represents a visible prefix ending:
        # an online terminal run can continue, while visible early stops cannot.
        prefix_length = 7
        for uptime in (1, 4):
            for initial in (0, 1):
                for prefix in product((0, 1), repeat=prefix_length):
                    schedule = list(prefix) + [prefix[-1]] * (HORIZON - prefix_length)
                    expected = forward_run_oracle(prefix, uptime, initial)
                    violations = violated_mt_rows(self.models[uptime, initial], schedule, initial)
                    with self.subTest(uptime=uptime, initial=initial, prefix=prefix):
                        self.assertEqual(not violations, expected, str(violations))


if __name__ == "__main__":
    unittest.main(verbosity=2)
