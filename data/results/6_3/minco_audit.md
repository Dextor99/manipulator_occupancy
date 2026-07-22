# Chapter 6.3 MINCO-risk audit

P1 and P3 MINCO-risk trajectories have identical q(t) and p_inner across the 10 frozen instances. Their final_risk is zero and final_cost is identical, while dense D_min varies with the independent GT obstacles. This supports the interpretation that these instances converge to the same risk-free low-cost local solution, not that J_smooth or plot_samples were reused incorrectly.

## A
- J unique count: 1
- J range: 3.69679157291 to 3.69679157291
- max q(t) abs diff vs first: 0
- max p_inner abs diff vs first: 0
- D_min range: 0.262566 to 0.330971
- final risk max: 0

## B
- J unique count: 5
- J range: 0.752037427445 to 3.69679157291
- max q(t) abs diff vs first: 1.12
- max p_inner abs diff vs first: 1.02
- D_min range: 0.00478902 to 0.315609
- final risk max: 4.71e-07

## C
- J unique count: 1
- J range: 3.69679157291 to 3.69679157291
- max q(t) abs diff vs first: 0
- max p_inner abs diff vs first: 0
- D_min range: 0.15494 to 0.214443
- final risk max: 0
