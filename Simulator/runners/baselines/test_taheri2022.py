"""Independent numerical checks of the Taheri reproduction's certificates.

Run: python -m Simulator.runners.baselines.test_taheri2022
The tests do not relax any production device model or change experiment files.
"""
from __future__ import annotations

import itertools
import math
import time
import unittest

import numpy as np
from scipy.optimize import linprog

from Simulator.runners.baselines import taheri_geometry as geometry
from Simulator.runners.baselines.taheri_data import ScenarioOracle


class TaheriGeometryChecks(unittest.TestCase):
    def test_two_dimensional_all_vertices_and_inner_ball(self):
        """Enumerate every possible polygon vertex, rather than sample normals."""
        A, b, _, metadata = geometry.lifted_ball(2, .1)
        result = geometry.fourier_motzkin(A, b, 2, time.monotonic()+10,
                                         max_rows=100000)
        self.assertEqual(result['status'], 'complete')
        E, d = result['A'], result['b']
        vertices = []
        for pair in itertools.combinations(range(len(d)), 2):
            rows = E[list(pair)]
            if abs(np.linalg.det(rows)) < 1e-12:
                continue
            point = np.linalg.solve(rows, d[list(pair)])
            if np.max(E@point-d) <= 1e-9:
                vertices.append(point)
        self.assertTrue(vertices)
        self.assertLessEqual(max(np.linalg.norm(v) for v in vertices), 1+1e-9)
        # A ball of radius r satisfies E_i x<=d_i exactly iff r||E_i||<=d_i.
        radius = metadata['inner_radius_bound']
        self.assertLessEqual(np.max(radius*np.linalg.norm(E, axis=1)-d), 1e-9)

    def test_projection_matches_lifted_lp(self):
        """Projection must preserve every tested support value of the lift."""
        A, b, _, _ = geometry.lifted_ball(3, .1)
        result = geometry.fourier_motzkin(A, b, 3, time.monotonic()+20,
                                         max_rows=150000)
        self.assertEqual(result['status'], 'complete')
        rng = np.random.default_rng(913)
        for direction in rng.normal(size=(12, 3)):
            lifted = linprog(np.r_[-direction, np.zeros(A.shape[1]-3)],
                             A_ub=A, b_ub=b, bounds=[(None,None)]*A.shape[1],
                             method='highs')
            projected = linprog(-direction, A_ub=result['A'], b_ub=result['b'],
                                bounds=[(None,None)]*3, method='highs')
            self.assertTrue(lifted.success and projected.success)
            self.assertAlmostEqual(lifted.fun, projected.fun, places=8)

    def test_ellipsoid_affine_identity_and_degenerate_stop(self):
        rng = np.random.default_rng(51)
        matrix = rng.normal(size=(5,5))
        Q, q, c = matrix.T@matrix+np.eye(5), rng.normal(size=5), -3.
        result = geometry.ellipsoid_to_ball(Q,q,c)
        points = rng.normal(size=(37,5))
        direct = np.einsum('ni,ij,nj->n',points,Q,points)+points@q+c
        sphere = np.sum((points@result['S'].T+result['offset'])**2,axis=1)-result['radius_squared']
        np.testing.assert_allclose(direct,sphere,atol=1e-10,rtol=1e-10)
        with self.assertRaises(ValueError):
            geometry.ellipsoid_to_ball(np.diag([1.,0.]),np.zeros(2),-1.)

    def test_farkas_fit_contains_in_target(self):
        G = geometry.market_template(2)
        # Bounded target square with a non-centered location also tests shift.
        E = np.vstack([np.eye(2),-np.eye(2)])
        d = np.array([1.2,.8,.4,.6])
        feasible = np.array([[.2,.1],[.3,.2],[.1,0.]])
        result = geometry.fit_homothet(G,feasible,E,d,time.monotonic()+15)
        self.assertIn('b',result)
        self.assertLessEqual(result['certificate_equality_residual'],1e-7)
        self.assertLessEqual(result['certificate_inequality_violation'],1e-7)
        for normal, limit in zip(E,d):
            support = linprog(-normal,A_ub=result['A'],b_ub=result['b'],
                              bounds=[(None,None)]*2,method='highs')
            self.assertTrue(support.success)
            self.assertLessEqual(-support.fun,limit+1e-7)

    def test_row_guard_preserves_remaining_auxiliary_variables(self):
        A,b,_,_=geometry.lifted_ball(8,.1)
        result=geometry.fourier_motzkin(A,b,8,time.monotonic()+5,max_rows=1)
        self.assertEqual(result['status'],'row_guard')
        self.assertGreater(result['remaining_auxiliaries'],0)
        np.testing.assert_array_equal(result['A'],A)


class ChanceLabelChecks(unittest.TestCase):
    def label_sequence(self,states):
        oracle=ScenarioOracle.__new__(ScenarioOracle)
        oracle.K=25; oracle.epsilon=.04; oracle.label_count=0
        iterator=iter(states)
        oracle.solve=lambda *args: {'state':next(iterator),'point':None,
                                    'objective':None,'lower_bound':None}
        return oracle.label(np.zeros(24),time.monotonic()+2)

    def test_one_unknown_never_turns_into_negative_label(self):
        result=self.label_sequence(['feasible']*23+['infeasible','unknown'])
        self.assertEqual(result['label'],0)
        self.assertEqual(result['scenarios_observed'],25)

    def test_threshold_24_of_25_and_two_certified_failures(self):
        self.assertEqual(self.label_sequence(['feasible']*24+['unknown'])['label'],-1)
        self.assertEqual(self.label_sequence(['infeasible']*2+['unknown']*23)['label'],1)


if __name__=='__main__':
    unittest.main(verbosity=2)
