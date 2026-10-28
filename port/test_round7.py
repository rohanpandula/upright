"""Opt-in output framing, independent of the geometry and damping. MIT."""
from pathlib import Path
import sys

import numpy as np
import pytest

import upright_port as up
from benchmark_adobe import agreement
from test_port import project_scene, W, H

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from validate_vs_adobe import CASES, SLOTS, parse_adobe


def test_native_slots_are_already_framed():
    for case in CASES.values():
        slots = parse_adobe(ROOT / 'adobe_truth' / f"img_{case['text']}.lua")
        for slot in SLOTS.values():
            np.testing.assert_allclose(up.frame_output(slots[slot]), slots[slot],
                                       rtol=0, atol=2e-9)


def test_translation_rule_and_shape_invariance():
    K = up._intrinsics_norm(1800, W, H)
    native = K @ up._rotation(-.35, .12, .04) @ np.linalg.inv(K)
    framed = up.frame_output(native)
    transform = framed @ np.linalg.inv(native / native[2, 2])
    np.testing.assert_allclose(transform[:2, :2], np.eye(2), atol=1e-15)
    np.testing.assert_allclose(transform[2], [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(up.frame_output(framed), framed, atol=1e-15)
    np.testing.assert_allclose(up.frame_output(-7*native), framed, atol=1e-15)
    translation = np.eye(3); translation[:2, 2] = [2, -3]
    np.testing.assert_allclose(up.frame_output(translation @ native), framed, atol=1e-14)
    assert agreement(native, framed, W, H)['shape_rms_pct'] < 1e-12
    assert agreement(native, framed, W, H)['raw_rms_pct'] > 1


@pytest.mark.parametrize('mode', ['auto', 'full', 'vertical', 'level', 'guided'])
def test_framing_follows_solve_and_damping_override(mode):
    segments, _, kind = project_scene(-20, 6, 3, 1800)
    if mode == 'guided':
        segments = np.concatenate([segments[kind == axis][:2] for axis in (0, 1)])
    original = up.solve_ex(segments, W, H, 1800, mode, damping=(.6, .7, .8))
    framed = up.solve_ex(segments, W, H, 1800, mode, damping=(.6, .7, .8), framing=True)
    assert original['ok'] and framed['ok']
    np.testing.assert_array_equal(framed['H_unframed'], original['H'])
    np.testing.assert_array_equal(framed['H'], up.frame_output(original['H']))
    for key in ('quality', 'rotation', 'damping', 'source_focal_px'):
        assert framed[key] == original[key]
    np.testing.assert_array_equal(up.solve(segments, W, H, 1800, mode,
        damping=(.6, .7, .8), framing=True), framed['H'])


def test_off_and_rejection_stay_identity():
    for mode in ('off', 'auto'):
        out = up.solve_ex(np.empty((0, 4)), W, H, 1800, mode, framing=True)
        np.testing.assert_array_equal(out['H'], np.eye(3))
        assert out['ok'] == (mode == 'off')


@pytest.mark.parametrize('bad', [np.eye(2), np.full((3, 3), np.nan),
    [[1, 0, 0], [0, 1, 0], [0, 0, 0]],
    [[1, 0, 0], [0, 1, 0], [-2, 0, 1]]])
def test_invalid_framing_is_rejected(bad):
    with pytest.raises(ValueError):
        up.frame_output(bad)
