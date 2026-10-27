"""Worker for the allocator ablation grid (separate module so multiprocessing 'spawn' can import it)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKEND = str(ROOT / "backend")


def run_one(args):
    case_key, assets, start, end, split, cond, seed = args
    sys.path.insert(0, BACKEND)
    import warnings
    warnings.filterwarnings("ignore")
    from app.backtest import engine
    from app.ml import allocator as alloc
    from app.data import market_data as md

    panel = md.get_multi_history(assets, start=start, end=end).dropna()

    orig_train = alloc.train_adaptive_policy

    def seeded_train(*a, **k):
        k["seed"] = seed
        return orig_train(*a, **k)

    alloc.train_adaptive_policy = seeded_train

    if cond == "no_regime":
        orig_dr = engine.detect_regimes

        def uniform_regimes(prices):
            r = orig_dr(prices)
            r.probabilities.loc[:, :] = 1.0 / 3.0   # strip all regime information from the state
            return r

        engine.detect_regimes = uniform_regimes

    gens = 0 if cond == "untrained" else 40
    res = engine.run_backtest(panel, primary_symbol="SPY", split_date=split, es_generations=gens, es_population=32)
    return dict(case=case_key, cond=cond, seed=seed, test=res["metrics_test_period"], full=res["metrics_full_period"])
