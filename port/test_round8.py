"""Destination scale policy, separate from geometry, damping and framing. MIT."""
from pathlib import Path
import json

import numpy as np
import pytest

import upright_port as up
from benchmark_adobe import agreement
from test_port import project_scene, W, H

ROOT = Path(__file__).resolve().parents[1]


def test_same_rotation_native_scale_recovery():
    rows = json.loads((ROOT/'evidence/round8.json').read_text())['rows']
    for r in rows:
        actual = up.scale_output(up.frame_output(np.array(r['base'])))
        limit = .0004 if r['group'] == 'four_scene' else .016
        # Retain the two independently known native framing failures.
        if r['name'] == 'V40S3' and r['mode'] in ('full', 'vertical'):
            limit = 4.
        assert agreement(r['H'], actual, 1400, 1000)['raw_rms_pct'] < limit


def test_scaling_preserves_angles_and_commutes_with_framing():
    K = up._intrinsics_norm(1800, W, H)
    for rotation in ((-.35, .12, .04), (.35, -.12, -.04), (0, 0, .3), (-.1, .2, -.1)):
        matrix = K @ up._rotation(*rotation) @ np.linalg.inv(K)
        matrix /= matrix[2, 2]
        scaled = up.scale_output(matrix)
        assert agreement(matrix, scaled, W, H)['shape_rms_pct'] < 1e-12
        np.testing.assert_allclose(up.scale_output(scaled), scaled, atol=1e-14)
        np.testing.assert_allclose(up.scale_output(-7*matrix), scaled, atol=1e-14)
        np.testing.assert_allclose(up.frame_output(scaled),
                                   up.scale_output(up.frame_output(matrix)), atol=1e-14)
        transform = scaled @ np.linalg.inv(matrix)
        np.testing.assert_allclose(transform[:2, :2], np.eye(2)*transform[0, 0], atol=1e-14)
        np.testing.assert_allclose(transform[2], [0, 0, 1], atol=1e-14)
    np.testing.assert_array_equal(up.scale_output(np.eye(3)), np.eye(3))


@pytest.mark.parametrize('mode', ['auto', 'full', 'vertical', 'level', 'guided'])
def test_opt_in_scale_follows_override_and_preserves_solve(mode):
    segments, _, kind = project_scene(-20, 6, 3, 1800)
    if mode == 'guided':
        segments = np.concatenate([segments[kind == axis][:2] for axis in (0, 1)])
    for framing in (False, True):
        original = up.solve_ex(segments, W, H, 1800, mode, framing=framing, damping=(.6, .7, .8))
        scaled = up.solve_ex(segments, W, H, 1800, mode, framing=framing,
                             scaling=True, damping=(.6, .7, .8))
        assert original['ok'] and scaled['ok']
        np.testing.assert_array_equal(scaled['H_unscaled'], original['H'])
        np.testing.assert_array_equal(scaled['H'], up.scale_output(original['H']))
        for key in ('rotation', 'damping', 'quality', 'source_focal_px'):
            assert scaled[key] == original[key]
        np.testing.assert_array_equal(up.solve(segments, W, H, 1800, mode,
            framing=framing, scaling=True, damping=(.6, .7, .8)), scaled['H'])
    for mode in ('off', 'auto'):
        out = up.solve_ex(np.empty((0, 4)), W, H, 1800, mode, framing=True, scaling=True)
        np.testing.assert_array_equal(out['H'], np.eye(3))
        assert out['ok'] == (mode == 'off')


@pytest.mark.parametrize('bad', [np.eye(2), np.full((3, 3), np.nan),
    [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 1, 0], [-2, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [0, 0, 1]]])
def test_invalid_scale_is_rejected(bad):
    with pytest.raises(ValueError):
        up.scale_output(bad)
