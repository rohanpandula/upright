"""Real detected supports and resolution-invariant quality policy regressions."""
from pathlib import Path
import unittest
import numpy as np
import upright_port as up


class TestRound10(unittest.TestCase):
    def test_real_photo_supports_accept_at_original_and_quarter_size(self):
        path=Path(__file__).resolve().parents[1]/'evidence/round10_regression_supports.npz'
        with np.load(path) as saved:
            for iid in ('2120337','3416332'):
                for scale in (1.,.25):
                    w,h=np.rint(saved[iid+'_size']*scale).astype(int)
                    seg=saved[iid+'_seg']*np.tile([w,h]/saved[iid+'_size'],2)
                    f=up.metadata_focal_px(float(saved[iid+'_focal_mm']),w,h)
                    for mode in ('auto','full'):
                        with self.subTest(image=iid,scale=scale,mode=mode):
                            out=up.solve_ex(seg,w,h,f,mode,framing=True,scaling=True)
                            self.assertTrue(out['ok'],(out['reason'],out['quality']))
                            self.assertFalse(np.array_equal(out['H'],np.eye(3)))

    def test_quality_scatter_and_bias_are_resolution_invariant(self):
        # Realistic symmetric .6-degree scatter: more evidence must not increase
        # the upper bias bound and make an unbiased family fail automatically.
        a=np.radians(np.tile(np.linspace(-1.2,1.2,81),3))
        vertical=np.column_stack([np.arange(len(a))*.001,np.zeros(len(a)),
                                  np.arange(len(a))*.001+.5*np.sin(a),.75*np.cos(a)])
        horizontal=np.column_stack([np.zeros(len(a)),np.arange(len(a))*.001,
                                    .75*np.cos(a),np.arange(len(a))*.001+.5*np.sin(a)])
        decisions=[]
        for w,h in ((1200,800),(4800,3200),(12000,8000)):
            q=up._alignment_quality(np.eye(3),vertical,horizontal,w,h)
            self.assertTrue(q['passed'],q)
            self.assertEqual(q['limit_deg'],1.)
            decisions.append(q)
        for q in decisions[1:]:
            for key in ('family_median_deg','family_signed_median_deg','family_median_margin_deg'):
                np.testing.assert_allclose(q[key],decisions[0][key],atol=1e-12)


if __name__=='__main__':unittest.main()
