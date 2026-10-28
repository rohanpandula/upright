"""Tests for the experimental selector, retained even if it cannot ship."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'port'),str(ROOT/'evidence')]
from test_port import project_scene,W,H
spec=importlib.util.spec_from_file_location('round12_candidate',ROOT/'evidence/round12_candidate_port.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)


def test_margin_requires_both_conditions_and_strict_absolute_difference():
    assert up._manhattan_margin(8,6)
    assert not up._manhattan_margin(8,6.000001)
    assert not up._manhattan_margin(4,3)
    assert not up._manhattan_margin(5,3.5)
    assert up._manhattan_margin(5,3.499)
    for a,b in [(0,0),(np.inf,1),(8,np.nan),(8,-1)]:
        assert not up._manhattan_margin(a,b)


def test_two_vp_gate_on_one_projected_world_family():
    seg,_,kind=project_scene(22,12,35,2200)
    assert up.estimate_vanishing_points(seg[kind==0],W,H)==[]
    out=up.solve_ex(seg[kind==0],W,H,2200,'full')
    assert not out['ok']
    np.testing.assert_array_equal(out['H'],np.eye(3))


def test_roll_and_perspective_preserve_world_support_identity():
    for roll in (-40,-25,25,40):
        seg,_,kind=project_scene(22,12,roll,2200)
        for seed in (0,1,2):
            vps=up.estimate_vanishing_points(seg,W,H,seed=seed)
            assert len(vps)>=2
            flat=np.concatenate([v['inliers'] for v in vps])
            assert len(np.unique(flat))==len(flat)
            for v in vps:v['vp']=v['vp']/[W,H,1]
            vertical,horizontal=up._classify(vps,seg/[W,H,W,H],W,H,2200)
            assert vertical is not None and horizontal
            assert np.all(kind[vertical['inliers']]==0)
            assert np.all(kind[horizontal[0]['inliers']]==1)


def test_equally_supported_competing_manhattan_frames_are_rejected():
    vps=[];parts=[];offset=0
    for roll in (-20,20):
        seg,R,kind=project_scene(12,8,roll,2200)
        K=up._intrinsics_norm(2200,W,H)
        for k,axis in [(0,[0,1,0]),(1,[1,0,0])]:
            idx=np.flatnonzero(kind==k)+offset
            vps.append(dict(vp=K@R.T@axis,inliers=idx,weight=10.))
        parts.append(seg);offset+=len(seg)
    vertical,horizontal=up._classify(vps,np.concatenate(parts)/[W,H,W,H],W,H,2200)
    assert vertical is None and horizontal==[]
