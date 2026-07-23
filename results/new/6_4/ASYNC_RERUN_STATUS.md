# 6.4 Strict Async Rerun Status

This directory contains a completed strict-asynchronous virtual-loop rerun.
It supersedes the earlier synchronous-result interpretation for claims about
asynchronous rolling replanning.

What was fixed:

- virtual time, obstacle motion, observations, and trajectory clock continue while planning is pending;
- candidates are planned for a fixed switch slot and are revalidated at that planned switch time;
- candidates that miss the switch deadline are rejected;
- task-safe success and replan-switch success are reported separately;
- online validation uses medium density, with dense GT recheck reported offline;
- optimizer convergence is recorded separately from candidate acceptance;
- bridge risk between submission and planned switch is logged.

Main finding:

- D1 body-crossing is mostly supported: CCRO-NUBS task-safe rate is 14/15 and replan-switch rate is 13/15.
- D2 end-effector crossing is not yet stable under the strict async timing: CCRO-NUBS task-safe rate is 7/15 and replan-switch rate is 7/15.
- D3 no-risk and D4 immediate-hold functional checks remain complete.

Paper-use guidance:

- Use this result as an honest strict-async capability-boundary result.
- Do not claim stable asynchronous dynamic replanning for all D1/D2 cases.
- A positive final-claim experiment still requires either stronger execution-layer safety correction or a method-independent operating-domain restriction based on trigger-to-contact lead time.
