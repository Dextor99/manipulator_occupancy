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

