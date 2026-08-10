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
predicted risk at 0.14 m while current clearance stayed above 0.12 m).

