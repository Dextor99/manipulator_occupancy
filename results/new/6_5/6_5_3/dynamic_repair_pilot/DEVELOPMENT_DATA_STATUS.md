# Development-data status

Trials D1 r02–r15 are retained as implementation-development and debugging
records. They are excluded from formal Section 6.5.3 statistics.

In particular, D1 r14 and its candidate playback are not valid repair or
execution evidence: the repair had `accepted_steps = 0`, and the historical
playback completion check could pass from its first feedback sample without
demonstrating motion. The current implementation requires a positive repair
step, measurable clearance improvement, measurable trajectory change, minimum
observation time, and measured departure from the starting joints.

D1 r16 is also a development record: the trigger and stop path executed, but the
repair candidate was rejected (`accepted_steps = 0`, zero clearance gain, zero
trajectory change, `fast_ms = 58`, 6 scale attempts). It demonstrates the
trigger→stop sequencing, not obstacle avoidance execution.

D1 r17 is a development record for the recorded-reference alignment and the
dynamic-track validity gate: the guard stopped the robot mid-stroke
(`GUIDED_GUARD_STOPPED_NO_CCRO_TRIGGER`), the reference completed only 475/980
index (TCP Y 0.40 → 0.013 m), two early frames showed clamped reference jumps,
and 245 armed frames were blocked as `predicted_track_not_dynamic`. The reference
never completed its stroke, so no trigger was attempted. It is not formal
evidence for Section 6.5.3.

The clean empty-workspace alignment audit is `reference_alignment_validation/r01`
(`REFERENCE_ALIGNMENT_PASS`): reference armed and tracked 967/980 index across
the full 0.4 → −0.4 m stroke, zero clamped steps, max joint match 0.0010 rad,
guard active throughout, no CCRO trigger.

`dynamic_track_audit/r01`–`r08` are development records of the foam-crossing
tracking audit; all eight `DYNAMIC_TRACK_AUDIT_FAIL`. None produced a track
meeting the qualifying gate (≥5 tracked frames, ≥0.15 m net displacement,
≥0.08 m/s window speed, ≥2 consecutive dynamically-valid frames). r01–r05 were
recorded with the earlier median-speed window and produced fragmented tracks
(r02: 8 rows/2 valid; r04: 33 rows/0 valid). r06–r08 used the timestamped
net-motion window with hysteresis; the closest attempt is r06 track 9, which
held ~0.1 m/s instant speed and a 0.121 m/s window peak but never latched
`dynamic_state` — the 5-sample window diluted net displacement across
missed-frame gaps, so `speed_ok` never held long enough for the two-frame valid
streak. r07 shows heavy fragmentation (19 distinct IDs, most single-row), i.e.
the foam cluster is not yet trackable as one compact object under the 0.03–0.10 m
geometry prefilter. These runs are not formal 6.5.3 evidence and gate the next
obstacle trial until a track qualifies.

After removing the radius geometry gate (radius became diagnostic, not an
object-identity filter) and feeding every external cluster to the dynamic
tracker, `dynamic_track_audit_final/r01` is the first `DYNAMIC_TRACK_AUDIT_PASS`:
track 25 held 80 tracked frames, 0.551 m net displacement, 0.111 m/s max window
speed, and 57 consecutive dynamically-valid frames (track 24: 11 frames, 0.063 m,
max run 6). This passes the frozen `cluster_eps=0.05` + temporal-denoise audit
and ends the stationary audit phase.

D1 r18 is a development `moving-shadow-stop` pilot taken after the audit
passed. It is not formal 6.5.3 evidence: the dynamic track (id 2, 251 frames,
0.388 m net, 0.081 m/s window peak) became dynamically valid for only 3 frames,
and those frames were blocked as `predicted_non_scene_link` — the closest point
was `left_link`/`gripper_base_link` (gripper finger), not the D1
upper-arm/forearm risk links. The raw-cloud hard guard then stopped the robot
at 0.085–0.099 m (`GUIDED_GUARD_STOPPED_NO_CCRO_TRIGGER`), which is correct
fail-safe behaviour but not the intended pilot outcome (the pilot wanted
predicted risk at 0.14 m while current clearance stayed above 0.12 m). r18
motivated the scene-independent protocol: `risk_links` no longer gate stops,
and the measured most-at-risk link decides.

D1 r19 is the first development `moving-shadow-stop` pilot under the unified
`653_unified_d1_d2_v1` protocol that reached `TRIGGERED`. Track 1 became
prediction-ready at frame 90 (0.081 m/s window), was blocked for 34 frames as
`future_reference_clearance_above_threshold` (predicted distance 0.18–0.30 m),
then at frame 124 `D_predicted = 0.128 m < 0.14 m` with `D_current = 0.238 m`
and `D_guard = 0.247 m` — exactly the intended evidence signature. The
immediate stop was sent ~76 ms after the trigger frame. The trial then produced
a `REJECTED_CANDIDATE`: `fast_elapsed_ms = 341.7 > 150 ms` budget,
`max_delta_q = 0`, `clearance_improvement = 0`, verification min distance
`−0.018 m` — the candidate is a reference continuation with no improving SCvx
step. Sensing → dynamic-track → predicted-risk → trigger → stop is now proven
end to end; the Fast local repair still fails to produce an acceptable
candidate within budget, so r19 is not formal 6.5.3 avoidance evidence.

D1 r20–r23 are further development `moving-shadow-stop` pilots taken after the
Fast-repair loop gained a hard wall-clock deadline (`deadline_perf`) and the
online acceptance switched to complete-pipeline time (repair + candidate
verification + reference verification + comparison) against the 150 ms budget.
r20 (`NO_TRIGGER`) never produced a prediction-ready frame — 73 armed frames
were all blocked `predicted_track_not_dynamic`, i.e. the track never reached the
0.08 m/s window-speed gate. r21–r23 all `TRIGGERED` (frames 145/120/108) and
stopped, but every candidate was `REJECTED_CANDIDATE` with `budget_exhausted =
True`: the deadline now bounds the loop (212–237 ms vs 341 ms in r19), yet the
dense distance scan on the candidate scale alone took 91–137 ms, so the budget
is exceeded before an accepted step. r23 came closest to acceptance:
`verification_min_distance = 0.104 m > 0.09 m` passed online clearance, but the
candidate was still a reference continuation (`max_delta_q = 0`,
`clearance_improvement = 0`) with no accepted SCvx step and a 237 ms pipeline.
The trigger→stop pipeline is repeatably proven; the SCvx step still cannot move
within budget, so none of r20–r23 are formal 6.5.3 avoidance evidence.

After the Fast repair added elastic tail-position variables (`z = [ΔQ, δq_T]`,
six tail variables in the finite sensitivity and QP, terminal velocity and
acceleration fixed to the reference), cheap linearized scale screening (only the
selected candidate gets the full dense geometric verification), and a bounded
1.25–2.00 s C2 rejoin-bridge search, `elastic_tail_replay` under r23 replays the
r23 trigger data offline and is the first evidence that an acceptable step can
be produced: `accepted_steps = 1`, elastic tail moved q1/q2 by −12.5/+23.7 mrad,
`fast_elapsed_ms = 105.9 < 150`, `verification_min_distance = 0.108 m > 0.09 m`,
`clearance_improvement = 3.3 mm ≥ 3 mm`, `max_delta_q = 28.6 mrad`. The sole
rejection is `safe_rejoin_not_found`: all four reference endpoints at 1.25–2.0 s
read 0.087/0.069/0.052/0.034 m (upperArm_Link, still inside the crossing
obstacle), so remaining stopped is the correct fail-closed decision rather than
a repair failure. This replay is development evidence for the repair mechanism,
not a formal avoidance execution.

D1 r24 is a development `moving-shadow-stop` pilot that adds the fresh post-stop
RGB-D recheck. It `TRIGGERED` at frame 144 and the recheck passed
(`post_stop_fresh_recheck.json`: 5 associated frames, 0.53 s span, fitted speed
0.129 m/s, conservative max radius 0.189 m), replacing the trigger-time obstacle
state for repair. The candidate was then `REJECTED_CANDIDATE`:
`fast_elapsed_ms = 173.9 > 150`, no accepted tail step (`tail_delta_q = 0`),
`verification_min_distance = −0.153 m` with the large conservative radius, and
`safe_rejoin_not_found`. The fresh-recheck infrastructure works; the live repair
still needs the elastic step to succeed on a 0.189 m-radius obstacle within
budget. r24 is not formal 6.5.3 avoidance evidence.

