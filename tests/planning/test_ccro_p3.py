import numpy as np
from experiments.p3_truth_io import evaluate_causal_cv

def test_causal_cv_is_exact_for_constant_velocity():
    t=np.arange(0,2.01,.05); c=np.column_stack((.5-.1*t,.2+.03*t,np.full_like(t,.4))); s=np.column_stack((t,c,np.full_like(t,.05)))
    result=evaluate_causal_cv(s,.5,.1,.001)
    assert result["samples"]>0 and result["rmse"]<1e-10 and result["coverage"]==1.0
