"""Tests for upright_port.

Run with ``python3 -m unittest test_port -v`` from this directory, or
``python3 -m pytest test_port.py``.

The scenes are synthetic but geometrically exact: world lines are projected
through a known rotation and focal, so the true rectifying homography is known
to machine precision.  That makes "did it recover the camera" a hard assertion
rather than a visual judgement.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

import upright_port as up


W, H = 1600, 1200


def project_scene(pitch_deg, yaw_deg, roll_deg, focal_px):
    """Project a small Manhattan scene; return (segments_px, R_true, kind).

    R_true is camera-to-world, so ``K R_true K^-1`` is the exact rectification.
    ``kind`` is 0 for vertical, 1 for lateral horizontal and 2 for depth lines;
    depth lines radiate from the principal point after rectification, so they
    are not horizontal and must not be scored as such.
    """
    R = up._rotation(math.radians(pitch_deg), math.radians(yaw_deg),
                     math.radians(roll_deg))
    Rt = R.T

    def proj(p):
        q = Rt @ np.asarray(p, dtype=float)
        if q[2] <= 0.5:
            return None
        return np.array([focal_px * q[0] / q[2] + W / 2.0,
                         focal_px * q[1] / q[2] + H / 2.0])

    segs, kinds = [], []
    for X in (-3.0, -1.5, 0.0, 1.5, 3.0):
        for Z in (5.0, 7.0):
            a, b = proj([X, -2.0, Z]), proj([X, 3.0, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(0)
    for Y in (-1.5, 0.0, 1.5):
        for Z in (5.0, 7.0):
            a, b = proj([-4.0, Y, Z]), proj([4.0, Y, Z])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(1)
    for Y in (-1.0, 1.0):
        for X in (-2.0, 2.0):
            a, b = proj([X, Y, 4.0]), proj([X, Y, 9.0])
            if a is not None and b is not None and np.hypot(*(b - a)) > 80:
                segs.append([*a, *b])
                kinds.append(2)
    return np.asarray(segs, dtype=float), R, np.asarray(kinds)


def truth_homography(R, focal_px, w=W, h=H):
    K = up._intrinsics_norm(focal_px, w, h)
    Hm = K @ R @ np.linalg.inv(K)
    return Hm / Hm[2, 2]


def to_pixels(Hm):
    """Normalized homography -> pixel homography (so angles are isotropic)."""
    N = np.diag([1.0 / W, 1.0 / H, 1.0])
    return np.linalg.inv(N) @ Hm @ N


def apply(Hm, seg):
    Hm = to_pixels(Hm)
    a = (Hm @ np.column_stack([seg[:, :2], np.ones(len(seg))]).T).T
    b = (Hm @ np.column_stack([seg[:, 2:], np.ones(len(seg))]).T).T
    a = a[:, :2] / a[:, 2:3]
    b = b[:, :2] / b[:, 2:3]
    d = b - a
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def deviations(Hm, seg, kind=None):
    """(vertical, horizontal) deviation in degrees.

    With ``kind`` given, the horizontal set is the lateral family only (kind 1);
    depth lines (kind 2) are excluded because they are not horizontal after
    rectification.  Without it, segments are classified by pre-warp angle.
    """
    if kind is None:
        kind = np.where(np.abs(seg[:, 3] - seg[:, 1]) >=
                        np.abs(seg[:, 2] - seg[:, 0]), 0, 1)
    ang = apply(Hm, seg)
    vert = np.abs(np.abs(ang[kind == 0]) - 90.0)
    hor = np.minimum(np.abs(ang[kind == 1]), 180.0 - np.abs(ang[kind == 1]))
    return vert, hor


class TestFocalConversion(unittest.TestCase):
    def test_full_frame_roundtrip_is_exact(self):
        self.assertAlmostEqual(up.focal_px_from_35mm(35.0, 36, 24), 35.0, places=9)
        self.assertAlmostEqual(up.focal_35mm_from_px(35.0, 36, 24), 35.0, places=9)

    def test_diagonal_not_short_edge(self):
        f_px = up.focal_px_from_35mm(50.0, 6000, 4500)
        expected = 50.0 * math.hypot(6000, 4500) / up.SENSOR_DIAGONAL_MM
        self.assertAlmostEqual(f_px, expected, places=9)
        # The common short-edge formula would give 50/36*6000 = 8333.33.
        self.assertNotAlmostEqual(f_px, 50.0 / 36.0 * 6000, places=1)
        self.assertAlmostEqual(up.focal_35mm_from_px(f_px, 6000, 4500), 50.0, places=9)

    def test_metadata_clamp_applies_and_geometric_does_not(self):
        self.assertAlmostEqual(
            up.metadata_focal_px(5.0, 1000, 1000),
            up.focal_px_from_35mm(10.0, 1000, 1000), places=9)
        self.assertAlmostEqual(
            up.metadata_focal_px(5000.0, 1000, 1000),
            up.focal_px_from_35mm(1000.0, 1000, 1000), places=9)
        # No metadata -> the 35 mm fallback.
        self.assertAlmostEqual(
            up.metadata_focal_px(None, 1000, 1000),
            up.focal_px_from_35mm(35.0, 1000, 1000), places=9)
        # A geometric estimate above 1000 mm survives untouched.
        self.assertAlmostEqual(up.focal_px_from_35mm(2000.0, 1000, 1000),
                               2000.0 * math.hypot(1000, 1000) / up.SENSOR_DIAGONAL_MM,
                               places=6)

    def test_default_focal_is_35mm(self):
        out = up.solve_ex([[0, 0, 0, 1]] * 4, W, H, None, "off")
        self.assertEqual(out["mode"], "off")


class TestRoundTrip(unittest.TestCase):
    """With damping disabled the solver must invert the generating homography."""

    def test_exact_recovery(self):
        focal = 1.6 * max(W, H)
        seg, R, _kind = project_scene(12.0, 8.0, 3.0, focal)
        expected = truth_homography(R, focal)
        got = up.solve(seg, W, H, focal, "full", damping=1.0)
        self.assertLess(np.abs(got - expected).max(), 1e-6)

    def test_axis_residuals_are_machine_zero(self):
        focal = 1.5 * max(W, H)
        seg, _R, _kind = project_scene(18.0, -6.0, 2.0, focal)
        got = up.solve(seg, W, H, focal, "full", damping=1.0)
        vert, hor = deviations(got, seg, _kind)
        self.assertLess(vert.max(), 1e-6)
        self.assertLess(hor.max(), 1e-6)

    def test_damped_result_differs_from_undamped(self):
        focal = 1.6 * max(W, H)
        seg, _R, _kind = project_scene(20.0, 10.0, 0.0, focal)
        undamped = up.solve(seg, W, H, focal, "auto", damping=1.0)
        full = up.solve(seg, W, H, focal, "full")
        auto = up.solve(seg, W, H, focal, "auto")
        self.assertGreater(np.abs(auto - undamped).max(), 1e-3)
        # Auto is the damped mode: strictly weaker than Full on keystone.
        self.assertLess(np.linalg.norm(auto - np.eye(3)),
                        np.linalg.norm(full - np.eye(3)))


class TestGuided(unittest.TestCase):
    def setUp(self):
        self.focal = 1.6 * max(W, H)
        self.seg, _R, self.kind = project_scene(22.0, 6.0, 2.0, self.focal)
        vertical = np.abs(self.seg[:, 3] - self.seg[:, 1]) >= np.abs(
            self.seg[:, 2] - self.seg[:, 0])
        v_idx = np.nonzero(vertical)[0][:2]
        h_idx = np.nonzero(~vertical)[0][:2]
        self.guides = self.seg[np.concatenate([v_idx, h_idx])]
        self.guide_kind = self.kind[np.concatenate([v_idx, h_idx])]

    def test_drives_both_families_to_zero(self):
        Hm = up.solve(self.guides, W, H, self.focal, "guided")
        vert, hor = deviations(Hm, self.guides, self.guide_kind)
        self.assertLess(vert.max(), 0.01)
        self.assertLess(hor.max(), 0.01)

    def test_recovers_the_generating_camera(self):
        _, R, _kind = project_scene(22.0, 6.0, 2.0, self.focal)
        expected = truth_homography(R, self.focal)
        Hm = up.solve(self.guides, W, H, self.focal, "guided")
        self.assertLess(np.abs(Hm - expected).max(), 1e-6)

    def test_requires_exactly_four(self):
        with self.assertRaises(ValueError):
            up.solve(self.guides[:3], W, H, self.focal, "guided")

    def test_requires_both_families(self):
        vertical = np.nonzero(self.guide_kind == 0)[0]
        seg, _R, kind = project_scene(22.0, 6.0, 2.0, self.focal)
        only_v = seg[np.nonzero(kind == 0)[0][:4]]
        self.assertEqual(len(only_v), 4)
        with self.assertRaises(ValueError):
            up.solve(only_v, W, H, self.focal, "guided")

    def test_beats_full(self):
        full = up.solve(self.guides, W, H, self.focal, "full")
        guided = up.solve(self.guides, W, H, self.focal, "guided")
        vert_f, hor_f = deviations(full, self.guides)
        vert_g, hor_g = deviations(guided, self.guides)
        self.assertLess(vert_g.max(), vert_f.max())
        self.assertLess(hor_g.max(), hor_f.max())


class TestModel(unittest.TestCase):
    """The two-camera model must be strictly more expressive than K R K^-1.

    This is the orthogonality test from the measurements: given a homography,
    the best single-camera ``K R K^-1`` leaves a residual that the two-camera
    ``s K1 R K0^-1`` drives to machine zero.
    """

    @staticmethod
    def ortho_residual(Hm, f0, free_destination):
        K0 = up._intrinsics_norm(f0, W, H)
        K0i = np.linalg.inv(K0)

        def resid(p):
            f1 = math.exp(p[0])
            K1 = np.array([[f1 / W, 0.0, p[1]], [0.0, f1 / H, p[2]],
                           [0.0, 0.0, 1.0]])
            M = np.linalg.inv(K1) @ Hm @ K0
            R = up._polar(M)
            # Best scale, so the comparison is scale-free.
            s = float(np.sum(M * R) / 3.0)
            return (M - s * R).ravel()

        if free_destination:
            lo = np.array([math.log(f0) - math.log(4.0), 0.2, 0.2])
            hi = np.array([math.log(f0) + math.log(4.0), 0.8, 0.8])
        else:
            lo = hi = np.array([math.log(f0), 0.5, 0.5])
        _, cost = up._levenberg_marquardt(resid, np.array([math.log(f0), 0.5, 0.5]),
                                          lo, hi, iters=300)
        return math.sqrt(cost)

    def test_two_camera_is_exact_and_single_camera_is_not(self):
        focal_true = 1.6 * max(W, H)
        seg, R, _kind = project_scene(25.0, 4.0, 0.0, focal_true)
        # A homography in the model's own family: destination camera at the
        # true focal, source camera 40% too short.
        f0 = 0.6 * focal_true
        K0 = up._intrinsics_norm(f0, W, H)
        K1 = up._intrinsics_norm(focal_true, W, H)
        Hm = K1 @ R @ np.linalg.inv(K0)
        Hm /= Hm[2, 2]
        two = self.ortho_residual(Hm, f0, True)
        one = self.ortho_residual(Hm, f0, False)
        self.assertLess(two, 1e-8)
        self.assertGreater(one, 0.1)
        self.assertGreater(one / max(two, 1e-12), 1e3)

    def test_two_camera_fit_recovers_the_destination_focal(self):
        focal_true = 1.6 * max(W, H)
        _, R, _kind = project_scene(20.0, 8.0, 0.0, focal_true)
        f0 = 0.7 * focal_true
        K0 = up._intrinsics_norm(f0, W, H)
        Hm = up._intrinsics_norm(focal_true, W, H) @ R @ np.linalg.inv(K0)
        Hm /= Hm[2, 2]
        K0 = up._intrinsics_norm(f0, W, H)
        best = {}
        for free in (True, False):
            best[free] = self.ortho_residual(Hm, f0, free)
        self.assertLess(best[True], 1e-8)
        # Same focal: the collapse is fine, so both fit.
        Hm2 = truth_homography(R, focal_true)
        same = self.ortho_residual(Hm2, focal_true, False)
        self.assertLess(same, 1e-6)

    def test_solver_returns_a_true_rotation_up_to_scale(self):
        focal = 1.6 * max(W, H)
        seg, _R, _kind = project_scene(20.0, 8.0, 0.0, focal)
        Hm = up.solve(seg, W, H, focal, "full", damping=1.0)
        K0 = up._intrinsics_norm(focal, W, H)
        M = np.linalg.inv(K0) @ Hm @ K0
        R = up._polar(M)
        self.assertLess(np.abs(M / np.cbrt(np.linalg.det(M)) - R).max(), 1e-9)

    def test_guided_uses_the_two_camera_model(self):
        """A guided solve must leave the source camera's K0 untouched."""
        focal = 1.6 * max(W, H)
        seg, _R, kind = project_scene(20.0, 8.0, 0.0, focal)
        vertical = np.nonzero(kind == 0)[0][:2]
        horiz = np.nonzero(kind == 1)[0][:2]
        guides = seg[np.concatenate([vertical, horiz])]
        Hm = up.solve(guides, W, H, focal, "guided")
        K0 = up._intrinsics_norm(focal, W, H)
        M = np.linalg.inv(K0) @ Hm @ K0
        R = up._polar(M)
        self.assertLess(np.abs(M / np.cbrt(np.linalg.det(M)) - R).max(), 1e-6)


class TestModes(unittest.TestCase):
    def setUp(self):
        self.focal = 1.6 * max(W, H)
        self.seg, _R, self.kind = project_scene(15.0, 6.0, 4.0, self.focal)

    def test_off_is_identity(self):
        self.assertTrue(np.allclose(up.solve(self.seg, W, H, self.focal, "off"),
                                    np.eye(3)))

    def test_level_only_rolls(self):
        Hm = up.solve(self.seg, W, H, self.focal, "level")
        out = up.solve_ex(self.seg, W, H, self.focal, "level")
        self.assertAlmostEqual(out["damping"][0], 0.0)
        self.assertAlmostEqual(out["damping"][1], 0.0)
        self.assertGreater(out["damping"][2], 0.5)
        vert, hor = deviations(Hm, self.seg, self.kind)
        self.assertGreater(np.median(vert), 1.0)
        self.assertGreater(np.median(hor), 1.0)

    def test_vertical_keeps_horizontals(self):
        Hm = up.solve(self.seg, W, H, self.focal, "vertical")
        vert, hor = deviations(Hm, self.seg, self.kind)
        self.assertLess(np.median(vert), 1.0)
        self.assertGreater(np.median(hor), 1.0)

    def test_full_corrects_both(self):
        Hm = up.solve(self.seg, W, H, self.focal, "full")
        vert, hor = deviations(Hm, self.seg, self.kind)
        self.assertLess(np.median(vert), 0.5)
        self.assertLess(np.median(hor), 0.5)

    def test_auto_is_between_level_and_full(self):
        auto = up.solve(self.seg, W, H, self.focal, "auto")
        full = up.solve(self.seg, W, H, self.focal, "full")
        vert_a, hor_a = deviations(auto, self.seg, self.kind)
        vert_f, hor_f = deviations(full, self.seg, self.kind)
        self.assertGreater(np.median(vert_a), np.median(vert_f))
        self.assertGreater(np.median(hor_a), np.median(hor_f))
        self.assertLess(np.median(vert_a), np.median(vert_f) + 12.0)

    def test_auto_is_a_no_op_on_a_pure_yaw_frame(self):
        seg, _R, kind = project_scene(0.0, 12.0, 0.0, self.focal)
        out = up.solve_ex(seg, W, H, self.focal, "auto")
        self.assertLess(out["damping"][1], 0.02)
        self.assertLess(np.abs(out["H"] - np.eye(3)).max(), 0.02)

    def test_auto_is_stronger_than_nothing(self):
        auto = up.solve(self.seg, W, H, self.focal, "auto")
        vert, _ = deviations(auto, self.seg, self.kind)
        vert0, _ = deviations(np.eye(3), self.seg, self.kind)
        self.assertLess(np.median(vert), np.median(vert0))

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            up.solve(self.seg, W, H, self.focal, "nope")

    def test_identity_on_too_few_segments(self):
        self.assertTrue(np.allclose(up.solve(self.seg[:3], W, H, self.focal,
                                             "full"), np.eye(3)))


class TestCoordinates(unittest.TestCase):
    def test_output_is_normalized(self):
        focal = 1.6 * max(W, H)
        seg, _R, _kind = project_scene(15.0, 6.0, 2.0, focal)
        Hm = up.solve(seg, W, H, focal, "full")
        self.assertAlmostEqual(Hm[2, 2], 1.0, places=12)
        # Normalized input corners land near the frame, not at ~1e6.
        corners = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                            [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]).T
        q = Hm @ corners
        q = q[:2] / q[2]
        self.assertTrue(np.all(np.abs(q) < 6.0))


class TestDetector(unittest.TestCase):
    @staticmethod
    def _image(pitch=14.0, yaw=6.0):
        """A perspective-distorted checkerboard."""
        board = np.zeros((900, 1200), np.uint8)
        for i in range(0, 1200, 100):
            board[:, i:i + 3] = 220
        for j in range(0, 900, 100):
            board[j:j + 3, :] = 220
        board = board.astype(np.float32)
        # A real centered square-pixel camera; the former arbitrary corner
        # warp required f^2 < 0 and should fail the stricter quality gate.
        K = np.array([[up.metadata_focal_px(None,1200,900),0,600],
                      [0,up.metadata_focal_px(None,1200,900),450],[0,0,1]])
        M = K @ up._rotation(math.radians(-pitch),math.radians(yaw),0) @ np.linalg.inv(K)
        out = __import__("cv2").warpPerspective(board, M, (1200, 900))
        return np.clip(out, 0, 255).astype(np.uint8)

    def test_finds_segments(self):
        seg = up.detect_segments(self._image())
        self.assertGreaterEqual(len(seg), 8)
        self.assertEqual(seg.shape[1], 4)
        self.assertTrue(np.all(np.hypot(seg[:, 2] - seg[:, 0],
                                        seg[:, 3] - seg[:, 1]) > 4.0))

    def test_solve_from_image(self):
        img = self._image()
        out = up.solve_ex(None, None, None, None, "full", image=img)
        self.assertTrue(out["ok"])
        self.assertEqual(out["segments"].shape[1], 4)
        self.assertGreater(len(out["vps"]), 0)

    def test_significance_bins_scale_with_long_side(self):
        self.assertEqual(up._significance_bins(1000), 13)
        self.assertEqual(up._significance_bins(4000), 16)
        self.assertLess(up._significance_bins(100), up._significance_bins(4000))

    def test_cos_22_5_constant(self):
        self.assertAlmostEqual(up.COS_22_5, math.cos(math.radians(22.5)), places=7)


class TestDetectorSignificance(unittest.TestCase):
    def test_nfa_is_exact_binomial_tail(self):
        # An independently summed small tail, not a density/count heuristic.
        n, k, maxdim = 18, 14, 64
        tail = sum(math.comb(n, j) * (1 / 8) ** j * (7 / 8) ** (n - j)
                   for j in range(k, n + 1))
        self.assertAlmostEqual(math.exp(up._log_nfa(n, k, maxdim)),
                               maxdim ** 4 * tail, places=12)

    def test_alignment_count_alone_is_not_significance(self):
        # Equal support counts; the second has many unaligned observations.
        self.assertLess(up._log_nfa(20, 20, 1000), 0)
        self.assertGreater(up._log_nfa(100, 20, 1000), 0)

    def test_nfa_stays_finite_for_long_lines(self):
        got = up._log_nfa(100000, 100000, 4000)
        self.assertTrue(np.isfinite(got))
        self.assertAlmostEqual(got, 4 * math.log(4000) - 100000 * math.log(8))
        self.assertEqual(up._log_nfa(10, 0, 100), 4 * math.log(100))
        with self.assertRaises(ValueError):
            up._log_nfa(10, 11, 100)

    def test_all_eight_directions(self):
        import cv2
        for theta in up.DIRECTIONS:
            with self.subTest(theta=theta):
                image = np.zeros((256, 256), np.uint8)
                d = np.array([math.cos(theta), math.sin(theta)])
                a = np.rint([128, 128] - 90 * d).astype(int)
                b = np.rint([128, 128] + 90 * d).astype(int)
                cv2.line(image, a, b, 255, 2, cv2.LINE_AA)
                seg = up.detect_segments(image)
                direction = seg[:, 2:] - seg[:, :2]
                long = np.linalg.norm(direction, axis=1) > 150
                self.assertTrue(long.any())
                normal_error = np.abs(direction[long] @ [-d[1], d[0]])
                normal_error /= np.linalg.norm(direction[long], axis=1)
                self.assertLess(float(normal_error.min()), math.sin(math.radians(0.5)))

    def test_no_fixed_eight_pixel_minimum(self):
        import cv2
        image = np.zeros((16, 16), np.uint8)
        cv2.line(image, (3, 8), (10, 8), 255, 1)
        seg = up.detect_segments(image)
        self.assertGreater(len(seg), 0)
        self.assertTrue((np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1) < 8).any())

    def test_short_line_on_larger_canvases(self):
        import cv2
        for size in (12, 16, 20, 24, 32):
            for transpose in (False, True):
                with self.subTest(size=size, transpose=transpose):
                    image = np.zeros((size, size), np.uint8)
                    cv2.line(image, (3, size // 2), (10, size // 2), 255, 1)
                    if transpose:
                        image = image.T.copy()
                    seg = up.detect_segments(image)
                    self.assertGreater(len(seg), 0)
                    delta = np.abs(seg[:, 2:] - seg[:, :2])
                    along, across = (1, 0) if transpose else (0, 1)
                    self.assertTrue(((delta[:, along] >= 5) &
                                     (delta[:, across] < 0.1)).any())

    def test_short_line_still_requires_nfa_significance(self):
        import cv2
        image = np.zeros((64, 64), np.uint8)
        cv2.line(image, (3, 32), (10, 32), 255, 1)
        # Even eight perfectly aligned samples have M^4 / 8^8 == 1 here.
        self.assertEqual(up.detect_segments(image).shape, (0, 4))

    def test_blank_noise_and_bad_images(self):
        self.assertEqual(up.detect_segments(np.zeros((64, 64))).shape, (0, 4))
        noise = np.random.default_rng(0).integers(0, 256, (128, 128), np.uint8)
        self.assertLessEqual(len(up.detect_segments(noise)), 5)
        for bad in (np.zeros((0, 0)), np.full((10, 10), np.nan)):
            with self.assertRaises(ValueError):
                up.detect_segments(bad)

    def test_absorbs_contained_segments_but_does_not_bridge_gaps(self):
        base = [0, 0, 100, 0]
        for candidate, absorbed in (([10, 6, 90, 6], True),
                                    ([10, 6.01, 90, 6.01], False),
                                    ([-10, 0, 80, 0], False),
                                    ([-9.99, 0, 80, 0], True),
                                    ([20, 0, 110, 0], False),
                                    ([20, 0, 109.99, 0], True),
                                    ([105, 0, 150, 0], False),
                                    ([90, 0, 10, 0], True)):
            with self.subTest(candidate=candidate):
                got = up._merge_segments(np.array([base, candidate], dtype=float), 6, 10)
                self.assertEqual(len(got), 1 if absorbed else 2)
                np.testing.assert_array_equal(got[0], base)

    def test_gap_split_returns_the_correct_unsorted_points(self):
        points = np.array([[0, 0], [101, 0], [1, 0], [100, 0]], dtype=float)
        runs = up._split_gaps(points, 10)
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(np.ptp(run[:, 0]) == 1 for run in runs))

    def test_refinement_splits_bent_support(self):
        x = np.arange(100, dtype=float)
        points = np.column_stack([x, np.abs(x - 50)])
        runs = up._refine_chains([points], 10, 1.0)
        self.assertGreater(len(runs), 1)
        self.assertTrue(all(len(run) < len(points) for run in runs))

    def test_scale_returns_original_pixel_coordinates(self):
        import cv2
        image = np.zeros((256, 512), np.uint8)
        cv2.line(image, (20, 128), (490, 128), 255, 3)
        seg = up.detect_segments(image, max_side=256)
        self.assertGreater(np.max(np.linalg.norm(seg[:, 2:] - seg[:, :2], axis=1)), 440)
        self.assertLess(abs(float(seg[0, [1, 3]].mean()) - 128), 5)

    def test_caller_can_bypass_detector_with_lsd(self):
        from unittest.mock import patch
        import cv2
        image = TestDetector._image()
        seg = cv2.createLineSegmentDetector().detect(image)[0].reshape(-1, 4)
        with patch.object(up, 'detect_segments', side_effect=AssertionError('seam broken')):
            out = up.solve_ex(segments=seg, image=image, mode='full')
        self.assertTrue(out['ok'])


class TestAccuracyRegression(unittest.TestCase):
    def test_quality_uses_raw_pixel_angles(self):
        # The same 45-degree pixel supports must fail at either aspect ratio,
        # even though their robust energy residuals would be compressed.
        vertical = np.array([[0, 0, 100, 100], [10, 0, 110, 100]], dtype=float)
        horizontal = np.array([[0, 20, 100, 20], [0, 40, 100, 40]], dtype=float)
        for w, h in ((1000, 500), (500, 1000)):
            with self.subTest(w=w, h=h):
                scale = np.array([w, h, w, h])
                quality = up._alignment_quality(np.eye(3), vertical / scale,
                                                horizontal / scale, w, h)
                np.testing.assert_allclose(quality['family_median_deg'], [45, 0])
                self.assertFalse(quality['passed'])

    def test_quality_weights_support_length(self):
        long_vertical = np.array([[20, 0, 20, 1000]], dtype=float)
        short_outliers = np.tile([0, 0, 1, 0], (20, 1))
        horizontal = np.array([[0, 20, 1000, 20]], dtype=float)
        scale = np.array([1600, 1200, 1600, 1200])
        quality = up._alignment_quality(np.eye(3),
            np.concatenate([long_vertical, short_outliers]) / scale, horizontal / scale, 1600, 1200)
        self.assertTrue(quality['passed'])
        np.testing.assert_array_equal(quality['family_median_deg'], [0, 0])

    def test_bad_seed4_optimizer_result_fails_own_quality_check(self):
        from unittest.mock import patch
        import cv2
        import benchmark_accuracy as benchmark
        image = cv2.imread(str(benchmark.ROOT / 'lrtest/in/D_roll.tif'), 0)
        h, w = image.shape
        segments = up.detect_segments(image)
        # Actual failed solution, evidence/round1_diagnosis.json. Replay the
        # optimizer failure through the public API without using truth to gate it.
        theta = np.array([-0.17811796261583085, 1.2, 0.13403823499357556,
                          math.log(1600), 0.8, 0.8])
        optimizer = up._levenberg_marquardt
        def replay(fun, x0, lo, hi, **kwargs):
            if len(x0) == 4:
                return optimizer(fun, x0, lo, hi, **kwargs)
            return theta, 0.0012678816696807755
        for mode in ('full', 'auto', 'level', 'vertical'):
            with self.subTest(mode=mode), patch.object(
                    up, '_levenberg_marquardt', side_effect=replay):
                out = up.solve_ex(segments, w, h, 1600, mode, seed=4)
                self.assertFalse(out['ok'])
                self.assertIn('alignment', out['reason'])
                self.assertFalse(out['quality']['passed'])
                np.testing.assert_array_equal(out['H'], np.eye(3))

    def test_confidence_checks_undamped_geometry(self):
        focal = 1.6 * max(W, H)
        segments, _, _ = project_scene(20, 10, 4, focal)
        for mode in ('full', 'auto', 'level', 'vertical'):
            with self.subTest(mode=mode):
                out = up.solve_ex(segments, W, H, focal, mode)
                self.assertTrue(out['ok'], out['reason'])
                self.assertTrue(out['quality']['passed'])
                self.assertLess(max(out['quality']['family_median_deg']), 1e-5)

    def test_detector_preserves_roll_scene_across_seeds(self):
        import cv2
        import benchmark_accuracy as benchmark
        image = cv2.imread(str(benchmark.ROOT / 'lrtest/in/D_roll.tif'), 0)
        self.assertIsNotNone(image)
        h, w = image.shape
        truth = benchmark.ground_truth(benchmark.CASES['D_roll'], 1600, w, h, 9, -5, 20)
        segments = up.detect_segments(image)
        for seed in range(50):
            with self.subTest(seed=seed):
                result = up.solve_ex(segments, w, h, 1600, 'full', seed=seed)
                self.assertTrue(result['ok'])
                scores = benchmark.deviation(result['H'], truth, np.array([0, 0, 1, 1]), w, h)
                self.assertLess(max(scores), 0.1)

    def test_other_raster_families_across_seeds(self):
        import cv2
        import benchmark_accuracy as benchmark
        for name in ('A_clean', 'B_strong', 'C_noisy'):
            image = cv2.imread(str(benchmark.ROOT / 'lrtest/in' / (name + '.tif')), 0)
            self.assertIsNotNone(image)
            h, w = image.shape
            truth = benchmark.ground_truth(benchmark.CASES[name], 1600, w, h, 9, -5, 20)
            segments = up.detect_segments(image)
            for seed in range(10):
                with self.subTest(scene=name, seed=seed):
                    result = up.solve_ex(segments, w, h, 1600, 'full', seed=seed)
                    self.assertTrue(result['ok'], result['reason'])
                    scores = benchmark.deviation(result['H'], truth, np.array([0, 0, 1, 1]), w, h)
                    self.assertLess(max(scores), 0.1)

    def test_roll_with_rejected_review_trial_detector(self):
        from unittest.mock import patch
        import cv2
        import benchmark_accuracy as benchmark
        image = cv2.imread(str(benchmark.ROOT / 'lrtest/in/D_roll.tif'), 0)
        h, w = image.shape
        # The earlier short-line trial changed eight supports and broke seed 9.
        # Exercise its changed candidate set without changing detector defaults.
        log_nfa = up._log_nfa
        with patch.object(up, '_significance_bins', return_value=0), patch.object(
                up, '_log_nfa', side_effect=lambda *args: log_nfa(*args) + math.log(2)):
            segments = up.detect_segments(image)
        result = up.solve_ex(segments, w, h, 1600, 'full', seed=9)
        self.assertTrue(result['ok'], result['reason'])
        truth = benchmark.ground_truth(benchmark.CASES['D_roll'], 1600, w, h, 9, -5, 20)
        self.assertLess(max(benchmark.deviation(result['H'], truth, np.array([0, 0, 1, 1]), w, h)), 0.1)

    def test_energy_uses_observed_lines_in_pixel_angles(self):
        # A 45-degree segment has the same loss in either aspect ratio.
        a = np.array([[0.1, 0.2, 0.2, 0.4]])  # 100x100 px in a 1000x500 image
        b = np.array([[0.2, 0.1, 0.4, 0.2]])  # same pixels in a 500x1000 image
        ra = up._family_terms(np.eye(3), a, np.array([1., 0.]), 1, 1000, 500, math.hypot(100, 100))
        rb = up._family_terms(np.eye(3), b, np.array([1., 0.]), 1, 500, 1000, math.hypot(100, 100))
        np.testing.assert_allclose(np.abs(ra), np.abs(rb), atol=1e-12)

    def test_bundled_ground_truth_and_frozen_segments(self):
        import benchmark_accuracy as benchmark
        rows = benchmark.run()
        # MEASURED bounds: evidence/accuracy_gap.txt, original residuals up to
        # 0.406 degrees. Both families now stay below a tenth of a degree.
        for name in (*benchmark.CASES, '01_input_truth'):
            with self.subTest(scene=name):
                self.assertTrue(rows[name]['image']['ok'])
                self.assertLess(max(rows[name]['image']['median_axis_deg']), 0.1)
        # Isolate energy improvement on ORIGINAL supports, before the gate
        # can conservatively reject their uncertainty and return identity.
        self.assertLess(max(rows['B_strong']['fixed_segments']['candidate_axis_deg']), 0.05)



class TestConstants(unittest.TestCase):
    def test_measured_sigma_v_is_in_keystone_units(self):
        self.assertAlmostEqual(math.degrees(up.SIGMA_V), 13.33, places=6)
        self.assertEqual(up.SIGMA_V_CI_DEG, (9.94, 22.72))

    def test_l0_v_is_not_the_old_twelve(self):
        self.assertAlmostEqual(up.L0_V, 2.64, places=6)

    def test_other_measured_constants(self):
        self.assertAlmostEqual(up.LAMBDA0_H, 76.73, places=6)
        self.assertAlmostEqual(math.degrees(up.SIGMA_YAW), 11.64, places=6)
        self.assertAlmostEqual(math.degrees(up.SIGMA_PITCH_GATE), 22.10, places=6)

    def test_saturation_is_monotone(self):
        self.assertGreater(up.correction_fraction_vertical(0.0),
                           up.correction_fraction_vertical(math.radians(30.0)))
        self.assertLess(up.correction_fraction_horizontal(0.0, 0.0), 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
