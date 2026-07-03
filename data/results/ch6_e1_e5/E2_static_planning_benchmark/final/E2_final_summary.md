# E2 Final Summary

## Completed Baselines

- RRT-Connect + smoothing
- Official TrajOpt/Tesseract
- CHOMP-style
- TrajOpt-style
- GPMP2-style
- MINCO-risk
- Ours CCRO-NUBS

## Notes

- Official TrajOpt/Tesseract is bound through the `tesseract-robotics` PyPI package and evaluated by the shared dense verifier.
- CHOMP/TrajOpt/GPMP2 are lightweight style reproductions; keep their `*-style` labels.
- The official TrajOpt run uses Tesseract joint-space optimization; E2 point-cloud risk is not injected into Tesseract and is applied only by the common verifier.
- All methods are evaluated with the same dense `TrajectoryVerifier`.
- Supplemental perturbation statistics use 10 perturbed obstacle point clouds for each P1/P2/P3 scene and coarse risk evaluation at 0.1 s; they support robustness analysis but do not replace the dense final acceptance table.
- Representative P2 time-series visualization is saved as `final/figures/fig_E2_P2_Dmin_curve.png`.
- P2 motion smoothness is additionally visualized by `final/figures/fig_E2_P2_joint_kinematics.png` and `final/figures/fig_E2_P2_Jsmooth_bar.png`.
- Supplemental `Ours-fast` keeps dense validation unchanged and passes P1/P2/P3; in P3 it reduces planning time from `23191.6 ms` to `4008.31 ms` with `D_min=0.0986413`.
- In P2, Ours CCRO-NUBS reaches `D_min=0.10162`, RRT reaches `D_min=0.0968866`, and CHOMP-style reaches `D_min=0.31853`.
- Internal `NUBS-base` and `NUBS-EEF-risk` should remain in ablation rather than the main external table.
