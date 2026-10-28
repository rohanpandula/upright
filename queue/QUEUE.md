# Queued agents (cap = 8 concurrent)

## RUNNING
- detector (line detector)          2a7c6220
- focal_prior                        187d44e2
- digest                             8d381500
- sigma_v_precise                    9cd8d65a
- non_manhattan                      7d49922d
- vp_clustering                      2fa0f884
- sensor_constant                    df11583b
- version_slots                      3e5aa507

## QUEUED — launch one per freed slot
1. sigma_h_precise      — dense yaw x pitch grid, refit sigma_yaw/sigma_gate
2. redteam_model        — adversarial test of H = s K1 R K0^-1 (esp. over-parameterisation confound)
3. verify_curves        — independent re-derivation of both weight curves
4. port                 — clean licence-clean reference implementation + tests
5. negpy_compare        — deep NegPy comparison, integration sketch
6. patent_map           — claim-by-claim exposure analysis
7. site_audit           — every number on the site vs evidence
8. report_audit         — internal contradictions, stale numbers
9. harness              — harden ImportSweep against the 7 known failure modes
10. guided_limits       — Guided on a second scene, plus degenerate cases
11. mode_semantics      — Level/Vertical on hard scenes
