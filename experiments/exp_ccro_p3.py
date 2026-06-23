"""P3 recorded-data audit, truth evaluation and shadow-log normalization."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np, yaml
from experiments.p3_truth_io import evaluate_causal_cv, load_truth_csv

ROOT=Path(__file__).resolve().parents[1]

def _audit(pattern: str) -> list[dict]:
    rows=[]
    for name in sorted(glob.glob(str(ROOT/pattern))):
        folder=Path(name); manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8")); files=sorted((folder/"frames").glob("frame_*.npz")); times=[]; joint_ok=True
        for file in files:
            with np.load(file) as data:
                times.append(float(data["timestamp"])); joint_ok &= data["joint_values"].ndim==1 and len(data["joint_values"])>=6
        valid=bool(times) and np.all(np.isfinite(times)) and (len(times)<2 or np.all(np.diff(times)>0))
        rows.append({"sequence":folder.name,"scene":manifest.get("scene"),"manifest_frames":manifest.get("frames"),"actual_frames":len(files),"timestamps_valid":bool(valid),"joints_valid":bool(joint_ok),"duration":None if len(times)<2 else times[-1]-times[0]})
    return rows

def _synthetic(cfg: dict) -> dict:
    s=cfg["synthetic"]; rng=np.random.default_rng(s["seed"]); dt=s["dt"]; t=np.arange(0,s["duration"]+1e-12,dt); profiles={}
    profiles["constant"] = np.column_stack((.45-.05*t, .05*np.ones_like(t), .55*np.ones_like(t)))
    profiles["stop"] = np.column_stack((.45-.08*np.minimum(t,2.5), .08*np.ones_like(t), .55*np.ones_like(t)))
    profiles["reverse"] = np.column_stack((.35+.06*np.abs(t-2.5), -.05*np.ones_like(t), .52*np.ones_like(t)))
    profiles["accelerate"] = np.column_stack((.48-.012*t*t, .02*t, .56*np.ones_like(t)))
    results={}; all_errors=[]; all_cover=[]; pred=cfg["prediction"]
    for name,center in profiles.items():
        measured=center+rng.normal(0,s["measurement_sigma"],center.shape); series=np.column_stack((t,measured,np.full(len(t),.05)))
        score=evaluate_causal_cv(series,pred["horizon"],pred["step"],pred["uncertainty_radius"]); results[name]=score
        if score["rmse"] is not None: all_errors.extend([score["rmse"]**2]*score["samples"]); all_cover.extend([score["coverage"]]*score["samples"])
    return {"profiles":results,"pooled_rmse":float(np.sqrt(np.mean(all_errors))),"pooled_coverage":float(np.mean(all_cover))}

def _shadow(source: Path) -> list[dict]:
    metrics=json.loads(source.read_text(encoding="utf-8")); rows=[]
    for scenario,item in metrics["scenarios"].items():
        for event in item["active"]["events"]:
            rows.append({"evidence":"programmatic_shadow","scenario":scenario,"trigger_time":event["submitted_timestamp"],"planning_ms":event["elapsed_ms"],"outcome":event["outcome"],"candidate_accepted":event["candidate_accepted"],"candidate_min_distance":event["candidate_min_distance"],"passive_min_distance":item["passive"]["min_actual_distance"],"hypothetical_active_min_distance":item["active"]["min_actual_distance"],"rejection_reasons":event["rejection_reasons"]})
        for event in item["active"]["safety_events"]:
            rows.append({"evidence":"programmatic_shadow","scenario":scenario,"trigger_time":event["timestamp"],"planning_ms":None,"outcome":"safety_hold","candidate_accepted":False,"candidate_min_distance":None,"passive_min_distance":item["passive"]["min_actual_distance"],"hypothetical_active_min_distance":item["active"]["min_actual_distance"],"rejection_reasons":[event["reason"]]})
    return rows

def run(config_path=ROOT/"config/ccro_p3.yaml"):
    config=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); output=ROOT/config["output_dir"]; output.mkdir(parents=True,exist_ok=True)
    audits=[]
    for kind,pattern in config["recording_groups"].items():
        for row in _audit(pattern): row["kind"]=kind; audits.append(row)
    valid_rate=float(np.mean([r["timestamps_valid"] and r["joints_valid"] for r in audits])) if audits else 0.
    recorded={k:json.loads((ROOT/v).read_text(encoding="utf-8")) for k,v in config["recorded_results"].items()}
    synthetic=_synthetic(config); truth=load_truth_csv(ROOT/config["truth_file"]); truth_scores={k:evaluate_causal_cv(v,config["prediction"]["horizon"],config["prediction"]["step"],config["prediction"]["uncertainty_radius"]) for k,v in truth.items()}
    shadow=_shadow(ROOT/config["programmatic_shadow"]); a=config["acceptance"]
    checks={"sequence_count":len(audits)>=a["minimum_recorded_sequences"],"timestamps":valid_rate>=a["minimum_timestamp_valid_rate"],"synthetic_rmse":synthetic["pooled_rmse"]<=a["synthetic_rmse_max"],"synthetic_coverage":synthetic["pooled_coverage"]>=a["synthetic_coverage_min"],"shadow_records":len(shadow)>0}
    metrics={"stage":"P3","recorded_sequences":len(audits),"timestamp_valid_rate":valid_rate,"audit":audits,"recorded_warning_summary":recorded,"synthetic_truth":synthetic,"real_truth":{"status":"available" if truth else "pending_real_truth","series":len(truth),"scores":truth_scores},"shadow_records":len(shadow),"checks":checks,"simulation_accepted":bool(all(checks.values())),"real_truth_complete":bool(truth_scores),"accepted":bool(all(checks.values()) and truth_scores)}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+"\n",encoding="utf-8"); (output/"shadow_log.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in shadow),encoding="utf-8"); (output/"recording_audit.json").write_text(json.dumps(audits,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    table=f"| item | value |\n|---|---:|\n| recorded sequences | {len(audits)} |\n| timestamp valid rate | {valid_rate:.4f} |\n| synthetic pooled RMSE / m | {synthetic['pooled_rmse']:.5f} |\n| synthetic coverage | {synthetic['pooled_coverage']:.4f} |\n| shadow records | {len(shadow)} |\n| real truth series | {len(truth)} |\n\nSimulation acceptance: **{'PASS' if metrics['simulation_accepted'] else 'FAIL'}**  \nReal-truth acceptance: **{'PASS' if metrics['accepted'] else 'PENDING'}**\n"
    (output/"table_p3.md").write_text(table,encoding="utf-8"); return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default=str(ROOT/"config/ccro_p3.yaml")); a=p.parse_args(); m=run(a.config); print(json.dumps({k:m[k] for k in ("simulation_accepted","real_truth_complete","accepted","recorded_sequences","timestamp_valid_rate")},indent=2))
if __name__=="__main__": main()
