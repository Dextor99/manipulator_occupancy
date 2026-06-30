| scenario | expected | A4 static D_min | A5 replan D_min | A6 executor D_min | A5 accepted | A6 below stop/s | A6 finished | A6 goal error | A6 control p95/ms | state holds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | accepted_replan | 0.0498404 | 0.0617994 | 0.0751585 | 1 | 0 | True | 0.00093226 | 6.27557 | 0 |
| B | no_trigger | 1.08211 | 1.08214 | 1.08214 | 0 | 0 | True | 0.000285257 | 0.894205 | 0 |
| C | accepted_replan | 0.0425275 | 0.0827952 | 0.0639246 | 1 | 0 | True | 0.000202981 | 6.95792 | 0 |
| D | safety_takeover | 0 | 0 | 0 | 0 | 1.05 | False | 1.9799 | 6.64494 | 0 |
