"""Experiment 3: leakage-controlled evaluation + embedding ablation of the pattern matcher.

(a) Blocked 5-fold CV over the reference corpus. Within each label, windows are sorted by start date and split
    into 5 contiguous time blocks; every training window that overlaps (in calendar time) any test window is
    purged, so no near-duplicate overlapping window leaks across the split.
(b) Leave-one-event-out: train on the other two events + normal (purged against the held-out event's dates),
    test whether the unseen crisis is flagged as crisis-like (predicted as ANY event class rather than normal).
Feature variants: trained Transformer encoder (re-trained on the training windows only), untrained (random)
Transformer, raw resampled shape, PCA-16 of the raw shape, 7 hand-crafted summary statistics. Classifier for all:
distance-weighted 8-NN.
Run:  backend/.venv/bin/python paper/scripts/exp_matcher.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.ml import embedding, event_matcher as em  # noqa: E402

RES = ROOT / "paper" / "results"
LABELS = ["dotcom", "gfc", "covid", "normal"]
EVENTS = ["dotcom", "gfc", "covid"]

corpus = em._build_corpus()
X = np.stack([w.resampled for w in corpus]).astype(np.float32)
y = np.array([w.label for w in corpus])
starts = np.array([pd.Timestamp(w.start).value for w in corpus])
ends = np.array([pd.Timestamp(w.end).value for w in corpus])
n = len(corpus)
print("corpus", n, {l: int((y == l).sum()) for l in LABELS}, flush=True)

# ---------- fold assignment: contiguous time blocks within each label ----------
N_FOLDS = 5
fold = np.zeros(n, dtype=int)
for lab in LABELS:
    idx = np.where(y == lab)[0]
    order = idx[np.argsort(starts[idx])]
    for f, chunk in enumerate(np.array_split(order, N_FOLDS)):
        fold[chunk] = f


def purge_mask(test_idx):
    """True for windows that do NOT overlap in time with any test window."""
    keep = np.ones(n, dtype=bool)
    for i in test_idx:
        keep &= ~((starts <= ends[i]) & (ends >= starts[i]))
    keep[test_idx] = False
    return keep


# ---------- feature variants ----------
def feats_transformer(train_idx, seed, trained=True):
    mean, std = embedding.fit_scaler(X[train_idx])
    if trained:
        enc = embedding.train_encoder(X[train_idx], mean, std, seed=seed)
    else:
        torch.manual_seed(seed)
        enc = embedding.ShapeEncoder().eval()
    return embedding.embed(enc, X, mean, std)


def feats_raw(train_idx, pca=False):
    flat = X.reshape(n, -1)
    mu, sd = flat[train_idx].mean(0), flat[train_idx].std(0) + 1e-6
    Z = (flat - mu) / sd
    if pca:
        Z = PCA(16, random_state=0).fit(Z[train_idx]).transform(Z)
    return Z


def feats_stats(train_idx):
    S = np.column_stack([X[:, -1, 0], X[:, :, 0].min(1), X[:, :, 0].max(1), X[:, :, 1].mean(1), X[:, :, 1].max(1),
                         X[:, :, 2].min(1), X[:, :, 2].mean(1)])
    return (S - S[train_idx].mean(0)) / (S[train_idx].std(0) + 1e-6)


VARIANTS = {
    "Transformer (trained)": lambda tr, s: feats_transformer(tr, s, True),
    "Transformer (untrained)": lambda tr, s: feats_transformer(tr, s, False),
    "Raw shape (120-d)": lambda tr, s: feats_raw(tr),
    "PCA-16 of raw shape": lambda tr, s: feats_raw(tr, pca=True),
    "Summary statistics (7-d)": lambda tr, s: feats_stats(tr),
}
SEEDS = {"Transformer (trained)": [0, 1, 2], "Transformer (untrained)": [0, 1, 2]}


def knn_predict(F, train_idx, test_idx):
    clf = KNeighborsClassifier(n_neighbors=8, weights="distance").fit(F[train_idx], y[train_idx])
    return clf.predict(F[test_idx])


def scores(y_true, y_pred):
    return dict(
        balanced_acc=float(balanced_accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        acc=float((y_true == y_pred).mean()),
        recall={l: float(((y_pred == l) & (y_true == l)).sum() / max((y_true == l).sum(), 1)) for l in LABELS},
        crisis_fpr=float(((y_true == "normal") & (y_pred != "normal")).sum() / max((y_true == "normal").sum(), 1)),
        crisis_recall=float(((y_true != "normal") & (y_pred != "normal")).sum() / max((y_true != "normal").sum(), 1)),
    )


out = {"corpus": {l: int((y == l).sum()) for l in LABELS}, "cv": {}, "loeo": {}}
cms = {}

# ---------- (a) blocked CV ----------
for name, fn in VARIANTS.items():
    per_seed = []
    for seed in SEEDS.get(name, [0]):
        preds = np.empty(n, dtype=object)
        for f in range(N_FOLDS):
            test_idx = np.where(fold == f)[0]
            train_idx = np.where(purge_mask(test_idx))[0]
            F = fn(train_idx, seed)
            preds[test_idx] = knn_predict(F, train_idx, test_idx)
        per_seed.append(scores(y, preds.astype(str)))
        if name == "Transformer (trained)" and seed == 0:
            cms["cv_transformer"] = confusion_matrix(y, preds.astype(str), labels=LABELS).tolist()
        print("CV", name, "seed", seed, {k: round(v, 3) for k, v in per_seed[-1].items() if not isinstance(v, dict)},
              flush=True)
    agg = {k: float(np.mean([s[k] for s in per_seed])) for k in per_seed[0] if not isinstance(per_seed[0][k], dict)}
    agg.update({k + "_sd": float(np.std([s[k] for s in per_seed])) for k in
                ("balanced_acc", "macro_f1", "crisis_fpr", "crisis_recall")})
    agg["recall"] = {l: float(np.mean([s["recall"][l] for s in per_seed])) for l in LABELS}
    agg["n_seeds"] = len(per_seed)
    out["cv"][name] = agg

# ---------- (b) leave-one-event-out ----------
for ev in EVENTS:
    test_idx = np.where(y == ev)[0]
    train_idx = np.where(purge_mask(test_idx))[0]
    out["loeo"][ev] = {"n_test": int(len(test_idx)), "n_train": int(len(train_idx))}
    for name, fn in VARIANTS.items():
        rates = []
        for seed in SEEDS.get(name, [0]):
            F = fn(train_idx, seed)
            pred = knn_predict(F, train_idx, test_idx)
            rates.append(float((pred != "normal").mean()))
        out["loeo"][ev][name] = dict(crisis_like_rate=float(np.mean(rates)), sd=float(np.std(rates)))
        print("LOEO", ev, name, round(np.mean(rates), 3), flush=True)

out["confusion"] = {"labels": LABELS, **cms}

# ---------- deployed matcher: embedding visualisation + example queries ----------
matcher = em.get_matcher()
emb = matcher.embeddings
pca = PCA(2, random_state=0).fit(emb)
tsne = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
np.savez(RES / "embeddings.npz", emb=emb, pca=pca.transform(emb), tsne=tsne, labels=y,
         lengths=np.array([w.length for w in corpus]), explained=pca.explained_variance_ratio_)

queries = [
    ("COVID-19 crash window", "SPY", "2020-01-15", "2020-05-15", "covid"),
    ("2008 GFC window", "SPY", "2008-01-02", "2009-06-30", "gfc"),
    ("Dot-com bust window", "QQQ", "2000-03-01", "2002-10-01", "dotcom"),
    ("Calm 2013-2015", "SPY", "2013-01-01", "2015-12-31", None),
    ("Recent 2026 window", "SPY", "2026-03-01", "2026-10-05", None),
]
out["queries"] = {}
for name, sym, a, b, tgt in queries:
    r = em.match_window(sym, a, b, target_event=tgt)
    ra = em.match_window(sym, a, b)  # auto-detect mode
    out["queries"][name] = dict(
        symbol=sym, start=a, end=b, target=tgt, days=r["window_trading_days"],
        targeted=dict(event=r["match"]["event_id"], matched=r["match"]["matched"], confidence=r["match"]["confidence"],
                      purity=r["match"]["cluster_purity"]),
        auto=dict(event=ra["match"]["event_id"], matched=ra["match"]["matched"], confidence=ra["match"]["confidence"]),
        stats=dict(ann_return=r["query_stats"]["annualized_return"], ann_vol=r["query_stats"]["annualized_volatility"],
                   max_dd=r["query_stats"]["max_drawdown"], regime_mix=r["query_stats"]["regime_mix"]),
        query_path=r["query_stats"]["cumulative_return_path"], ref_path=r["reference"]["cumulative_return_path"],
        ref_window=[r["reference"]["window_start"], r["reference"]["window_end"], r["reference"]["symbol"]],
        fwd21=r["baseline"]["forward_return_21d_mean"],
    )
    print("Q", name, out["queries"][name]["targeted"], out["queries"][name]["auto"], flush=True)
out["matcher"] = dict(fit_method=matcher.fit_method, n_clusters=len(matcher.cluster_meta),
                      label_scales={k: float(v) for k, v in matcher.label_scales.items()})
json.dump(out, open(RES / "matcher.json", "w"), indent=1)
print("saved")
