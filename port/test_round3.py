"""Round 3 per-seed tails, VP/support consistency and gate discrimination. MIT."""
import unittest
from unittest.mock import patch
import numpy as np
import upright_port as up
import benchmark_accuracy as bench
import benchmark_quality as audit
import make_heldout_images as heldout


class TestRound3(unittest.TestCase):
    def test_vps_fit_their_returned_supports(self):
        image,_,_,_=heldout.render(heldout.CASES[8])
        segments=up.detect_segments(image)
        N=np.diag([1/1024,1/576,1])
        for seed in (0,1,2,4):
            out=up.solve_ex(segments,1024,576,heldout.camera(heldout.CASES[8])[1][0,0],
                            'full',seed=seed)
            self.assertTrue(out['ok'])
            for vp in out['vps']:
                # H09 has exactly three selected families. Their robust fits
                # must be stationary on the final supports they actually use.
                with self.subTest(seed=seed,count=len(vp['inliers'])):
                    pixels=np.linalg.inv(N)@vp['vp']
                    expected=up._refine_vp(segments[vp['inliers']],1024,576,pixels)
                    self.assertLess(np.linalg.norm(np.cross(up._normalize(pixels),expected)),1e-7)

    def test_all_heldout_seeds_have_small_tails(self):
        result=bench.run_heldout(range(50))
        for name,row in result['scenes'].items():
            for trial in row['trials']:
                with self.subTest(scene=name,seed=trial['seed']):
                    self.assertTrue(trial['ok'],trial['reason'])
                    self.assertLess(max(trial['median_axis_deg']),.1)
        self.assertLess(result['worst_family_deg'],.05)

    def test_input_focal_seed_tails(self):
        import cv2
        image=cv2.imread(str(bench.ROOT/'01_input.png'),0)
        h,w=image.shape
        segments=up.detect_segments(image)
        truth=bench.ground_truth((-.38,.10,.04),1400,w,h,8,-5,18)
        for focal in (1400,980,1200,1820):
            for seed in range(50):
                with self.subTest(focal=focal,seed=seed):
                    out=up.solve_ex(segments,w,h,focal,'full',seed=seed)
                    self.assertTrue(out['ok'],out['reason'])
                    self.assertLess(max(bench.deviation(out['H'],truth,
                        np.array([0,0,1,1]),w,h)),.1)
                    self.assertLess(abs(out['source_focal_px']/1400-1),.03)

    def test_gate_rejects_small_systematic_axis_bias(self):
        # The old gate accepts .11/.13 deg at these sizes; neither is a
        # four-pixel-scale optimizer catastrophe. Test both axes and aspects.
        for w,h in ((1200,800),(1024,576),(640,960)):
            vertical=np.array([[x,40,x,h-40] for x in (100,200,300,400)],float)
            horizontal=np.array([[40,y,w-40,y] for y in (100,200,300,400)],float)
            for angle in (.11,.13):
                a=np.radians(angle)
                H=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
                N=np.diag([1/w,1/h,1])
                q=up._alignment_quality(N@H@np.linalg.inv(N),vertical/[w,h,w,h],
                                         horizontal/[w,h,w,h],w,h)
                self.assertFalse(q['passed'],(w,h,angle,q))

    def test_gate_distinguishes_scatter_from_systematic_bias(self):
        # Symmetric noisy supports are accurate as an ensemble. Rotating
        # that SAME evidence by .13 degrees must fail, despite identical MAD.
        w,h=1200,800
        angles=np.tile(np.linspace(-.16,.16,41),3)
        a=np.radians(angles)
        vertical=np.column_stack([np.arange(len(a))*5,np.zeros(len(a)),
                                  np.arange(len(a))*5+600*np.sin(a),600*np.cos(a)])
        horizontal=np.array([[30,y,1100,y] for y in (100,300,500)],float)
        for bias,passed in ((0,True),(.13,False)):
            a=np.radians(bias);N=np.diag([1/w,1/h,1])
            H=N@np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])@np.linalg.inv(N)
            q=up._alignment_quality(H,vertical/[w,h,w,h],horizontal/[w,h,w,h],w,h)
            self.assertEqual(q['passed'],passed,q)

    def test_gate_rejects_original_h09_tail_candidates(self):
        image,_,_,_=heldout.render(heldout.CASES[8])
        segments=up.detect_segments(image)
        optimizer=up._levenberg_marquardt
        # Actual pre-fix bad undamped rotations, not an invented catastrophic fit.
        rotations=[(-.3101752936256536,.4399098759421401,.2608512822771301),
                   (-.31060776195229217,.4394460087806279,.26097880940477064)]
        for seed,rotation in enumerate(rotations):
            def replay(fun,x0,lo,hi,**kwargs):
                if len(x0)==4:return optimizer(fun,x0,lo,hi,**kwargs)
                return np.r_[rotation,x0[3:]],0.
            with self.subTest(seed=seed),patch.object(up,'_levenberg_marquardt',side_effect=replay):
                out=up.solve_ex(segments,1024,576,heldout.camera(heldout.CASES[8])[1][0,0],
                                'full',seed=seed)
            self.assertFalse(out['ok'])
            self.assertFalse(out['quality']['bias_passed'])
            self.assertTrue(out['quality']['scatter_passed'])
            np.testing.assert_array_equal(out['H'],np.eye(3))

    def test_impossible_camera_warp_is_rejected(self):
        import cv2
        board=np.zeros((900,1200),np.uint8)
        for x in range(0,1200,100):board[:,x:x+3]=220
        for y in range(0,900,100):board[y:y+3,:]=220
        M=cv2.getPerspectiveTransform(np.float32([[0,0],[1200,0],[1200,900],[0,900]]),
            np.float32([[120,90],[1080,60],[1150,840],[60,810]]))
        image=cv2.warpPerspective(board.astype(np.float32),M,(1200,900)).astype(np.uint8)
        segments=cv2.createLineSegmentDetector().detect(image)[0].reshape(-1,4)
        for supports in (None,segments):
            with self.subTest(detector='bundled' if supports is None else 'lsd'):
                out=up.solve_ex(supports,image=image,mode='full')
                self.assertFalse(out['ok'])
                self.assertFalse(out['quality']['bias_passed'])
                np.testing.assert_array_equal(out['H'],np.eye(3))

    def test_confusion_counts_rejected_candidates_not_identity(self):
        trials=[dict(gate_pass=p,candidate_axis_deg=[e,0])
                for p,e in ((True,.01),(True,.11),(False,.01),(False,.11))]
        trials.append(dict(gate_pass=False,candidate_axis_deg=None))
        self.assertEqual(audit.confusion(trials),dict(pass_good=1,pass_bad=1,
            reject_good=1,reject_bad=1,no_fit=1))


if __name__=='__main__':
    unittest.main()
