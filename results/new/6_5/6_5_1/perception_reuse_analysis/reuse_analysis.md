# Revised 6.5.1 Perception Reuse Analysis

This audit checks whether existing real RGB-D recordings/results can support the revised perception-only 6.5.1.

## Reuse Decision

Can reuse existing evidence: **True**

Usable evidence:
- real RealSense recordings with AUBO joint states exist
- empty/static decoupling results cover robot self-filtering and static obstacle detection
- dynamic approach/recover results cover obstacle tracking, STRO warning and risk recovery
- whole-body risk replay covers end-effector and middle-link nearest-link evidence

Gaps:
- no dedicated revised-6.5.1 static S1/S2/S3 dataset with elbow/forearm/wrist labels and 3 repeats each
- existing dynamic trials are reused from Chapter 4.3/4.4 and are not named D1/D2 with path markers
- existing metrics summarize object-level warning and whole-body risk separately; a unified per-frame 6.5.1 log may still be useful

Recommendation: Use existing results as preliminary/reuse evidence for 6.5.1, then optionally add a minimal perception-only recording set: E0 x2 poses, S1-S3 x3 static trials, D1-D2 x5 dynamic trials.

## Source Inventory

| item | source | frames/trials | real RGB-D | robot state | 6.5.1 use |
| --- | --- | --- | --- | --- | --- |
| E0 empty robot | data/results/ch4_2/scene_A_delta_0p050.json | 373 frames | yes | yes | self-filter/static detection |
| Static single obstacle | data/results/ch4_2/scene_B_delta_0p050.json | 377 frames | yes | yes | self-filter/static detection |
| Static multi obstacle | data/results/ch4_2/scene_B2_delta_0p050.json | 381 frames | yes | yes | self-filter/static detection |
| Static safe obstacle | data/results/ch4_3/final_static_A/metrics.json | 1 trials | yes | yes | STRO warning/recovery |
| Dynamic approach | data/results/ch4_3/final_dynamic/metrics.json | 8 trials | yes | yes | STRO warning/recovery |
| Approach-hold-leave | data/results/ch4_3/final_recover/metrics.json | 3 trials | yes | yes | STRO warning/recovery |
| CCRO ee risk | data/results/ch4_4/final_ee | 8 trials | yes | yes | nearest link / full-body risk |
| CCRO body risk | data/results/ch4_4/final_body_09_11 | 3 trials | yes | yes | nearest link / full-body risk |

## Static / Self-Filtering Evidence

| scene | method | frames | R_det | R_keep | R_over | sigma_c | T_dec(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E0 empty robot | Ours self-filter | 373 | - | - | - | - | 6.5001 |
| Static single obstacle | Ours self-filter | 377 | 1.0000 | 1.0000 | 0.0000 | 0.0168 | 6.2874 |
| Static multi obstacle | Ours self-filter | 381 | 1.0000 | 0.9999 | 0.0001 | 0.0132 | 6.2365 |

## Dynamic Warning / Recovery Evidence

| scene | trials | T_lead(s) | R_miss | R_false_time | D_trigger_ref(m) | T_recover(s) | N_switch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Static safe obstacle | 1 | - | - | 0.0100 | 0.2075 | - | 6.0000 |
| Dynamic approach | 8 | 5.1286 | 0.0000 | 0.4482 | 0.3551 | 0.8488 | 14.2500 |
| Approach-hold-leave | 3 | 1.2808 | 0.0000 | 0.0950 | 0.2376 | 0.3732 | 6.3333 |

## Whole-Body Risk / Nearest-Link Evidence

| scene | trials | sampled frames | EE risk frames | body risk frames | top nearest links |
| --- | --- | --- | --- | --- | --- |
| ee risk | 8 | 140 | 27 | 0 | left_link:58, gripper_base_link:57, upperArm_Link:13, wrist1_Link:11 |
| body risk | 3 | 185 | 0 | 118 | wrist1_Link:109, upperArm_Link:71, wrist2_Link:3, gripper_base_link:2 |
