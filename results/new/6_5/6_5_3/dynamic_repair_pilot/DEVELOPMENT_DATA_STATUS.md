# Development-data status

Trials D1 r02–r15 are retained as implementation-development and debugging
records. They are excluded from formal Section 6.5.3 statistics.

In particular, D1 r14 and its candidate playback are not valid repair or
execution evidence: the repair had `accepted_steps = 0`, and the historical
playback completion check could pass from its first feedback sample without
demonstrating motion. The current implementation requires a positive repair
step, measurable clearance improvement, measurable trajectory change, minimum
observation time, and measured departure from the starting joints.

