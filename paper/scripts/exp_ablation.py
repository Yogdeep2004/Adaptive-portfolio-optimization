"""Experiment 2: multi-seed evaluation and ablation of the adaptive allocator.
Conditions: full (regime + momentum + vol state), no_regime (regime probabilities replaced by a constant),
untrained (random-initialised policy, 0 ES generations). 6 seeds x 4 cases x 3 conditions.
Run:  backend/.venv/bin/python paper/scripts/exp_ablation.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp_workers import run_one  # noqa: E402

CASES = {
    "dotcom_deployed": (["SPY", "QQQ", "GC=F"], "1999-06-01", "2003-12-31", "2000-03-10"),
    "dotcom_corrected": (["SPY", "QQQ", "VUSTX"], "1999-06-01", "2003-12-31", "2000-03-10"),
    "gfc": (["SPY", "QQQ", "GLD", "TLT"], "2006-01-01", "2010-12-31", "2007-10-09"),
    "covid": (["SPY", "QQQ", "GLD", "TLT"], "2019-01-01", "2021-12-31", "2020-02-19"),
}
SEEDS = range(6)
CONDS = ["full", "no_regime", "untrained"]

if __name__ == "__main__":
    jobs = [(k, *v, cond, seed) for k, v in CASES.items() for cond in CONDS for seed in SEEDS]
    results = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i, r in enumerate(ex.map(run_one, jobs), 1):
            results.append(r)
            print(f"{i}/{len(jobs)} {r['case']} {r['cond']} seed={r['seed']} "
                  f"test_sharpe={r['test']['adaptive']['sharpe_ratio']}", flush=True)
    out = ROOT / "paper" / "results" / "ablation.json"
    json.dump(results, open(out, "w"), indent=1)
    print("saved", out)
