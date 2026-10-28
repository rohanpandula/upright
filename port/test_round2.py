"""Round 2 regressions. MIT; Python 3.12."""
import unittest
import cv2
import numpy as np
import benchmark_accuracy as bench
import upright_port as up
import sys
sys.path.insert(0, str(bench.ROOT))
import make_heldout_images as heldout


class TestSourceFocal(unittest.TestCase):
    def test_wrong_metadata_on_real_raster(self):
        image = cv2.imread(str(bench.ROOT / '01_input.png'), 0)
        h, w = image.shape
        segments = up.detect_segments(image)
        truth = bench.ground_truth((-.38, .10, .04), 1400, w, h, 8, -5, 18)
        for focal in (980, 1200, 1820):
            with self.subTest(focal=focal):
                out = up.solve_ex(segments, w, h, focal, 'full')
                self.assertTrue(out['ok'], out['reason'])
                self.assertLess(max(bench.deviation(out['H'], truth,
                    np.array([0, 0, 1, 1]), w, h)), .061668)
                self.assertLess(abs(out['source_focal_px'] / 1400 - 1), .03)

    def test_focal_damping_override_uses_recovered_source(self):
        image = cv2.imread(str(bench.ROOT / '01_input.png'), 0)
        a = up.solve_ex(image=image, focal_px=980, mode='full')
        b = up.solve_ex(image=image, focal_px=980, mode='full', damping=1)
        np.testing.assert_allclose(a['H'], b['H'], atol=1e-10)
        # The angle objective cannot estimate destination translation. Keep its
        # chosen centre fixed instead of drifting to an arbitrary bound.
        h,w=image.shape
        K=up._intrinsics_norm(a['source_focal_px'],w,h)
        expected=K@up._rotation(*a['rotation'])@np.linalg.inv(K)
        np.testing.assert_allclose(a['H'],expected/expected[2,2],atol=1e-9)

    def test_unobservable_frontal_focal_keeps_prior(self):
        # Perfect parallel families provide no finite focal constraint.
        vertical = np.array([[x, 40, x, 700] for x in (100, 300, 500, 700)], float)
        horizontal = np.array([[40, y, 900, y] for y in (100, 250, 400, 550)], float)
        focal, info = up._source_calibration(vertical/[960,720,960,720],
            horizontal/[960,720,960,720], np.array([0,1,0.]),
            np.array([1,0,0.]), 960,720,800)
        self.assertEqual(focal,800)
        self.assertFalse(info['accepted'])


class TestHeldout(unittest.TestCase):
    def test_wrong_focal_on_heldout_rasters(self):
        for case in heldout.CASES:
            image,truth,kind,_=heldout.render(case)
            w,h=case['w'],case['h']
            segments=up.detect_segments(image)
            focal=heldout.camera(case)[1][0,0]
            for factor in (.7,1.3):
                with self.subTest(scene=case['name'],factor=factor):
                    out=up.solve_ex(segments,w,h,focal*factor,'full')
                    self.assertTrue(out['ok'],out['reason'])
                    self.assertLess(max(bench.deviation(out['H'],truth,kind,w,h)),.15)

    def test_camera_coverage_and_visible_world_oracle(self):
        self.assertEqual({c['focal_mm'] for c in heldout.CASES}, {24,35,50,85})
        self.assertEqual({c['w']/c['h'] for c in heldout.CASES}, {3/2,4/3,16/9})
        self.assertEqual((min(c['pitch'] for c in heldout.CASES),
                          max(c['pitch'] for c in heldout.CASES)), (5,40))
        for c in heldout.CASES:
            with self.subTest(scene=c['name']):
                image,truth,kind,H = heldout.render(c)
                self.assertEqual(image.shape,(c['h'],c['w']))
                self.assertTrue(np.all(np.isfinite(truth)))
                self.assertGreater(np.sum(kind==0),2)
                self.assertGreater(np.sum(kind==1),2)
                self.assertLess(max(bench.deviation(H,truth,kind,c['w'],c['h'])),1e-8)
                if c['single_wall']:
                    self.assertFalse(np.any(kind==2))

    def test_heldout_image_pipeline_and_separate_scores(self):
        result = bench.run_heldout()
        self.assertEqual(result['attempts'],len(heldout.CASES))
        self.assertEqual(result['failures'],0)
        self.assertTrue(set(result['scenes']).isdisjoint(bench.CASES))
        # Lock down the independent renderer/detector path, including the wall
        # and near-frontal controls; no mixing with the fixture headline mean.
        for row in result['scenes'].values():
            self.assertLess(max(row['trials'][0]['median_axis_deg']),.15)
        scores=[t['median_axis_deg'] for r in result['scenes'].values() for t in r['trials']]
        self.assertAlmostEqual(result['mean_family_deg'],float(np.mean(scores)))

    def test_rejected_heldout_solves_are_counted_and_scored(self):
        from unittest.mock import patch
        with patch.object(up,'solve_ex',return_value=dict(H=np.eye(3),ok=False,
                reason='injected rejection',source_focal_px=None,calibration=None,quality=None)):
            result=bench.run_heldout()
        self.assertEqual(result['failures'],result['attempts'])
        for row in result['scenes'].values():
            np.testing.assert_allclose(row['uncorrected_axis_deg'],
                                       row['trials'][0]['median_axis_deg'])


if __name__ == '__main__':
    unittest.main()
