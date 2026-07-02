# E1 Final Summary

## Key Findings

- In `approach`, Ours-STRO reaches `T_lead=3.6280s`, while current-frame occupancy reaches `T_lead=1.1320s`.
- In `crossing`, Ours-STRO reaches `T_lead=4.6800s`, while current-frame occupancy reaches `T_lead=2.4800s`.
- The conservative prediction trade-off is visible in `R_false_time`: Ours-STRO is higher than current-frame methods in dynamic scenes.
- In `dynamic_future`, Critical-point APF detects future-only risk at `R_future=0.7647`, while Ours-CCRO Mesh reaches `R_future=1.0000`.

## Recommended Thesis Use

- Use `table_E1_occupancy_final.md` as the main risk/occupancy table.
- Use `table_E1_whole_body_apf_final.md` as the main whole-body APF comparison table.
- Use `fig_E1_dynamic_warning.png` and `fig_E1_whole_body_apf.png` as the two core E1 figures.
- Keep the earlier EEF-only and Body-current table as auxiliary evidence or E5 ablation.
