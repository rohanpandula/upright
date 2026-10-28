"""Adobe fidelity: response units, physical damping and shared mode decisions. MIT."""
import math
import unittest

import cv2
import numpy as np

import upright_port as up
from benchmark_accuracy import ROOT
from test_port import project_scene, W, H
from benchmark_adobe import agreement


class TestRound4(unittest.TestCase):
    def test_agreement_retains_framing_and_roll_separately(self):
        N = np.diag([1/W, 1/H, 1.])
        framing = N @ np.array([[1.3, 0, 80], [0, 1.3, -60], [0, 0, 1.]]) @ np.linalg.inv(N)
        score = agreement(np.eye(3), framing, W, H)
        self.assertGreater(score['raw_rms_pct'], 1)
        self.assertLess(score['shape_rms_pct'], 1e-12)
        roll = N @ up._rotation(0, 0, math.radians(2)) @ np.linalg.inv(N)
        self.assertGreater(agreement(np.eye(3), roll, W, H)['shape_rms_pct'], .5)
        for gauge in (7, -7):
            self.assertLess(agreement(np.eye(3), gauge*np.eye(3), W, H)['raw_rms_pct'], 1e-12)

    def test_measured_curve_uses_keystone_units(self):
        for angle in (0, 5, 15, 30, 40):
            lam = 2.64 * math.exp(-angle**2 / (2 * 13.33**2))
            self.assertAlmostEqual(up.correction_fraction_vertical(math.radians(angle)),
                                   lam / (1 + lam), places=12)
        self.assertEqual(up.SIGMA_V_CI_DEG, (9.94, 22.72))
        self.assertAlmostEqual(math.degrees(up.SIGMA_YAW), 11.64)

    def test_auto_removes_measured_fraction_of_visible_convergence(self):
        # Round 13: the response is keyed on undamped camera pitch, and the solver
        # must still deliver exactly that fraction of the detected line convergence.
        for pitch in (-34, -20, 12, 28):
            seg, _, kind = project_scene(pitch, 0, 0, 1800)
            out = up.solve_ex(seg, W, H, 1800, 'auto')
            self.assertTrue(out['ok'], out['reason'])
            vertical = seg[kind == 0] / [W, H, W, H]
            before = up._vertical_convergence(vertical, W, H)
            after = up._vertical_convergence(vertical, W, H, out['H'])
            fraction = up.correction_fraction_pitch(out['rotation'][0])
            self.assertAlmostEqual(1 - after / before, fraction, places=5)

    def test_pitch_curve_is_the_measured_corpus_shape(self):
        # Non-monotone: a low-angle shoulder, a peak near 8 deg, a collapse by 40 deg.
        curve = [up.correction_fraction_pitch(math.radians(a))
                 for a in (1, 4, 8, 12, 20, 30, 40)]
        self.assertLess(curve[0], curve[1])
        self.assertGreater(curve[2], curve[4])
        self.assertGreater(curve[4], curve[6])
        self.assertLess(curve[6], 0.02)
        self.assertEqual(up.correction_fraction_pitch(math.radians(41)), 0.0)
        self.assertGreaterEqual(min(curve), 0.0)
        self.assertLessEqual(max(curve), 1.0)
        # 12 deg is the port's own scene-A detected keystone at a true 22 deg pitch.
        self.assertGreater(up.correction_fraction_pitch(math.radians(22)), 0.4)
        self.assertLess(up.correction_fraction_pitch(math.radians(34)), 0.15)
        self.assertLess(up.correction_fraction_pitch(math.radians(40)), 0.02)

    def test_modes_share_the_undamped_candidate_and_gate(self):
        for name, f35 in (('A_clean', 38.9696), ('C_noisy', 39.7882)):
            image = cv2.imread(str(ROOT / 'lrtest' / 'in' / (name + '.tif')), 0)
            h, w = image.shape
            segments = up.detect_segments(image)
            focal = up.focal_px_from_35mm(f35, w, h)
            full = up.solve_ex(segments, w, h, focal, 'full')
            self.assertTrue(full['ok'], full['reason'])
            for mode in ('auto', 'level', 'vertical'):
                with self.subTest(scene=name, mode=mode):
                    out = up.solve_ex(segments, w, h, focal, mode)
                    self.assertTrue(out['ok'], out['reason'])
                    np.testing.assert_array_equal(out['rotation'], full['rotation'])
                    self.assertEqual(out['quality'], full['quality'])


if __name__ == '__main__':
    unittest.main()
