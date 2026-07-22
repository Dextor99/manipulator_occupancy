# Chapter 6.3 MINCO-risk audit

P1 and P3 MINCO-risk trajectories have identical q(t) and p_inner across the 10 frozen instances. Their initial_cost values vary by instance and are not equal to final_cost; the fixed-budget adapted MINCO-risk search nevertheless converges to the same final zero-risk trajectory in P1 and P3. Dense D_min varies with the independent GT obstacles, supporting the interpretation that J_smooth or plot_samples were not reused incorrectly.

## A
- J unique count: 1
- J range: 3.69679157291 to 3.69679157291
- max q(t) abs diff vs first: 0
- max p_inner abs diff vs first: 0
- D_min range: 0.262566 to 0.330971
- initial cost range: 2.74963 to 8.86969
- initial equals final count: 0/10
- final risk max: 0

## B
- J unique count: 5
- J range: 0.752037427445 to 3.69679157291
- max q(t) abs diff vs first: 1.12
- max p_inner abs diff vs first: 1.02
- D_min range: 0.00478902 to 0.315609
- initial cost range: 0.0979271 to 1.23321
- initial equals final count: 0/10
- final risk max: 4.71e-07

## C
- J unique count: 1
- J range: 3.69679157291 to 3.69679157291
- max q(t) abs diff vs first: 0
- max p_inner abs diff vs first: 0
- D_min range: 0.15494 to 0.214443
- initial cost range: 5.85652 to 15.4809
- initial equals final count: 0/10
- final risk max: 0
