"""Truth annotation I/O and causal constant-velocity evaluation for P3."""

from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

TRUTH_COLUMNS = ("sequence", "timestamp", "x", "y", "z", "radius", "source")


def load_truth_csv(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        return {}
    grouped: dict[str, list[list[float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(TRUTH_COLUMNS)-set(reader.fieldnames or [])
        if missing: raise ValueError(f"truth CSV missing columns: {sorted(missing)}")
        for row in reader:
            grouped.setdefault(row["sequence"], []).append([float(row[k]) for k in ("timestamp","x","y","z","radius")])
    result={}
    for key, rows in grouped.items():
        values=np.asarray(sorted(rows,key=lambda x:x[0]),float)
        if len(values)>1 and np.any(np.diff(values[:,0])<=0): raise ValueError(f"timestamps not strictly increasing: {key}")
        result[key]=values
    return result


def evaluate_causal_cv(series: np.ndarray, horizon: float, step: float, radius: float, history: int = 5) -> dict:
    """Evaluate predictions only against future samples available in truth."""
    errors=[]; covered=[]
    times=series[:,0]; centers=series[:,1:4]
    for i in range(1,len(series)):
        start=max(0, i-max(int(history), 2)+1)
        window_t=times[start:i+1]
        relative_t=window_t-window_t[-1]
        design=np.column_stack((relative_t, np.ones_like(relative_t)))
        velocity=np.linalg.lstsq(design, centers[start:i+1], rcond=None)[0][0]
        for tau in np.arange(step,horizon+1e-12,step):
            target = times[i] + tau
            if target > times[-1] + 1e-12:
                continue
            j=int(np.argmin(np.abs(times-target)))
            if j<=i or abs(times[j]-target)>max(step*.51,1e-3): continue
            error=float(np.linalg.norm(centers[i]+velocity*tau-centers[j])); errors.append(error); covered.append(error<=radius)
    if not errors: return {"samples":0,"rmse":None,"p95":None,"coverage":None}
    values=np.asarray(errors)
    return {"samples":len(errors),"rmse":float(np.sqrt(np.mean(values**2))),"p95":float(np.percentile(values,95)),"coverage":float(np.mean(covered))}
