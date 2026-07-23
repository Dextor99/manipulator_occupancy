# 6.4 Async Rerun Status

This directory currently contains an interrupted strict-asynchronous rerun.
It should not be used as the final paper result table.

Implemented corrections:

- virtual time continues while candidate planning is pending;
- candidate completion is delayed and revalidated at switch time;
- `switch_delay` is treated as a deadline, with actual switch after candidate completion;
- D1/D2 obstacles use `motion_start_time` and `pre_motion_center` to avoid t=0-only triggering;
- online validation is medium-density and dense GT recheck remains offline;
- optimizer convergence metadata is stored separately from candidate acceptance.

Current finding:

- corrected D1 smoke tests can pass after increasing dynamic risk weight;
- strict asynchronous formal rerun exposed failing D1/D2 instances and was interrupted;
- therefore the current strict-asynchronous result is not yet paper-ready.

Next required work:

- add asynchronous feasibility screening during D1/D2 instance generation, or improve the dynamic optimizer so delayed switch candidates keep sufficient margin;
- then rerun the full D1/D2 90-trial comparison plus D3/D4 functional checks;
- regenerate `metrics.json`, `manifest.json`, paper tables, and dense GT audit.
