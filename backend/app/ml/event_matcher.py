"""
Historical pattern comparison: does the SHAPE of a user-selected window
resemble the dot-com bust, the 2008 GFC, or the COVID crash -- or none of
them?

This is a different job from regime.py's HMM (which labels each individual
day bull/bear/crisis for the whole series). Here the unit of comparison is
one whole window, of any length, and the question is "which historical
episode's trend shape does this look like, if any."

Pipeline: build a reference corpus of many labeled windows (of varying
lengths, so lengths as different as COVID's ~33-day crash and the dot-com
bust's ~2.5 years are all represented), resample each to a fixed-length
shape via embedding.py, embed with a small Transformer encoder, group the
embeddings with HDBSCAN (sklearn's implementation -- no extra native
dependency) to check the corpus forms coherent shape groups, and score a
new query window against the labeled corpus points themselves by k-nearest
neighbour (_score_label).

Scoring deliberately does NOT go through cluster centroids. HDBSCAN builds
arbitrary-shaped density clusters, so a cluster's MEAN can land in empty
space no member occupies -- a query sitting exactly on top of real members
would then measure several "radii" away and score ~0 -- and ~16% of the
corpus is HDBSCAN noise belonging to no cluster at all, invisible to any
centroid. Distance to the actual nearest labeled windows has neither
failure mode.
"""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from . import embedding
from .regime import detect_regimes
from .. import config
from ..data import market_data as md
from ..data import synthetic_fallback

CORPUS_WINDOW_LENGTHS = [10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 110, 130,
                          160, 180, 220, 250, 320, 400, 500, 600]
OVERLAP_MATCH_THRESHOLD = 0.50
OVERLAP_DROP_THRESHOLD = 0.05
CRISIS_COVERAGE_THRESHOLD = 0.80
MAX_ENVELOPE_RATIO = 4.0
EVENT_CLASS_CAP = 120
NORMAL_CAP_RATIO = 2.5
MIN_QUERY_DAYS = 20
FORWARD_RETURN_DAYS = 21
CLUSTER_PURITY_THRESHOLD = 0.6
DISTANCE_SLACK = 1.4
CONFIDENCE_THRESHOLD = 0.20
CORPUS_SEED = 7
LOCAL_PURITY_K = 15
# How many same-label reference windows a query is scored against, and how
# many same-label neighbours define the in-class spacing that score is
# measured in. See _label_scales / _score_label.
MATCH_K = 8

CALM_REFERENCE_PERIODS = [
    ("SPY", "2003-06-01", "2006-12-31"),
    ("SPY", "2012-01-01", "2015-12-31"),
    ("SPY", "2016-06-01", "2019-11-01"),
    ("QQQ", "2012-01-01", "2015-12-31"),
]

EVENT_LABELS = {
    "dotcom": "Dot-com / Internet Boom & Bust",
    "gfc": "2008 Global Financial Crisis",
    "covid": "COVID-19 Crash & Recovery",
    "normal": "Normal / Typical Market",
}


@dataclass
class CorpusWindow:
    symbol: str
    label: str          # "dotcom" | "gfc" | "covid" | "normal"
    start: str
    end: str
    length: int                      # original trading-day length, pre-resample
    resampled: np.ndarray            # (40, 3) fixed-shape representation
    forward_return_21d: float | None


@dataclass
class Matcher:
    encoder: "embedding.ShapeEncoder"
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    corpus: list[CorpusWindow]
    embeddings: np.ndarray           # (n, 16)
    cluster_meta: dict               # cluster_id -> stats (see _build_cluster_meta)
    fit_method: str                  # "hdbscan_centroid" | "label_centroid_fallback"
    label_scales: dict               # label -> in-class spacing (see _label_scales)


_MATCHER: Matcher | None = None
_MATCHER_LOCK = threading.Lock()


def _majority_and_purity(labels: list[str]) -> tuple[str, float]:
    counts = Counter(labels)
    label, count = counts.most_common(1)[0]
    return label, count / len(labels)


def _slice_windows(series: pd.Series, symbol: str, event_id: str | None,
                    crisis_start: str | None, crisis_end: str | None) -> list[CorpusWindow]:
    """
    Slice `series` (a date-indexed price Series) into overlapping windows at
    every length in CORPUS_WINDOW_LENGTHS, labeling each by its overlap with
    [crisis_start, crisis_end]. If crisis_start is None, every window is
    unconditionally labeled "normal" (used for the calm reference periods,
    which have no crisis sub-range at all).

    A window earns the event label two ways: the crisis fills at least
    OVERLAP_MATCH_THRESHOLD of the WINDOW, or the window contains
    essentially the whole CRISIS (>= CRISIS_COVERAGE_THRESHOLD of it) while
    staying within MAX_ENVELOPE_RATIO times the crisis's own length. Without
    that second route a short crisis can only ever produce short windows --
    COVID's crash ran ~34 trading days, so no window longer than ~68 days
    could reach 50% overlap, and the whole class topped out at 60 days while
    dot-com and the GFC (multi-year declines) reached 600. Users routinely
    select several months around a crash, and that shape then had no
    same-length exemplar to match against. The ratio cap is what keeps this
    from swinging the other way and labelling a 600-day window "covid" on
    the strength of a 34-day crash buried inside it.
    """
    windows: list[CorpusWindow] = []
    n = len(series)
    crisis_start_ts = pd.Timestamp(crisis_start) if crisis_start else None
    crisis_end_ts = pd.Timestamp(crisis_end) if crisis_end else None
    crisis_days = (
        int(((series.index >= crisis_start_ts) & (series.index <= crisis_end_ts)).sum())
        if crisis_start_ts is not None else 0
    )

    for length in CORPUS_WINDOW_LENGTHS:
        if length + 5 >= n:
            continue
        stride = max(3, length // 5)
        for start_idx in range(0, n - length, stride):
            window = series.iloc[start_idx:start_idx + length]

            if crisis_start_ts is None:
                label = "normal"
            else:
                mask = (window.index >= crisis_start_ts) & (window.index <= crisis_end_ts)
                overlap = int(mask.sum())
                overlap_fraction = overlap / length
                crisis_coverage = overlap / crisis_days if crisis_days else 0.0
                envelopes_crisis = (
                    crisis_coverage >= CRISIS_COVERAGE_THRESHOLD
                    and length <= MAX_ENVELOPE_RATIO * crisis_days
                )
                if overlap_fraction >= OVERLAP_MATCH_THRESHOLD or envelopes_crisis:
                    label = event_id
                elif overlap_fraction <= OVERLAP_DROP_THRESHOLD:
                    label = "normal"
                else:
                    continue  # ambiguous transition window -- drop rather than mislabel

            fwd_idx = start_idx + length - 1 + FORWARD_RETURN_DAYS
            forward_return = None
            if fwd_idx < n:
                forward_return = float(series.iloc[fwd_idx] / series.iloc[start_idx + length - 1] - 1.0)

            channels = embedding.build_channels(window)
            resampled = embedding.resample_to_fixed(channels)
            windows.append(CorpusWindow(
                symbol=symbol, label=label,
                start=str(window.index[0].date()), end=str(window.index[-1].date()),
                length=length, resampled=resampled, forward_return_21d=forward_return,
            ))
    return windows


def _build_case_study_series(case: dict) -> tuple[str, pd.Series]:
    """Mirrors main.py's /api/backtest fallback pattern: live data, else synthetic."""
    # Use the case study's designated "shape" symbol (its most distinctive,
    # headline move -- see config.py) rather than always assets[0], so the
    # corpus and any query built from the same case study id stay consistent.
    symbol = case.get("shape_symbol", case["assets"][0])
    try:
        hist = md.get_history(symbol, start=case["start"], end=case["end"])
        prices = hist["adjclose"].dropna()
        if prices.empty:
            raise md.MarketDataError(f"empty history for {symbol}")
    except md.MarketDataError:
        dates = pd.bdate_range(case["start"], case["end"])
        panel = synthetic_fallback.generate_case_study_panel(case["id"], dates)
        prices = panel[symbol] if symbol in panel.columns else panel.iloc[:, 0]
    return symbol, prices


def _stratified_cap(pool: list[CorpusWindow], cap: int, rng: np.random.Generator) -> list[CorpusWindow]:
    """
    Subsample `pool` down to ~`cap` items, capping each distinct window
    LENGTH bucket roughly evenly rather than randomly subsampling the flat
    pool (which would just keep whichever length happens to be most
    numerous -- see the comment at the call site).
    """
    if len(pool) <= cap:
        return pool
    by_length: dict[int, list[CorpusWindow]] = {}
    for w in pool:
        by_length.setdefault(w.length, []).append(w)
    n_lengths = len(by_length)
    per_length_cap = max(cap // n_lengths, 3)

    kept: list[CorpusWindow] = []
    for length, items in by_length.items():
        if len(items) > per_length_cap:
            idx = rng.choice(len(items), size=per_length_cap, replace=False)
            kept.extend(items[i] for i in idx)
        else:
            kept.extend(items)

    if len(kept) > cap:
        idx = rng.choice(len(kept), size=cap, replace=False)
        kept = [kept[i] for i in idx]
    return kept


def _build_corpus(seed: int = CORPUS_SEED) -> list[CorpusWindow]:
    rng = np.random.default_rng(seed)
    windows_by_label: dict[str, list[CorpusWindow]] = {}

    for case in config.CASE_STUDIES.values():
        symbol, series = _build_case_study_series(case)
        ws = _slice_windows(series, symbol, case["id"], case["crisis_start"], case["crisis_end"])
        for w in ws:
            windows_by_label.setdefault(w.label, []).append(w)

    for symbol, start, end in CALM_REFERENCE_PERIODS:
        try:
            hist = md.get_history(symbol, start=start, end=end)
            series = hist["adjclose"].dropna()
        except md.MarketDataError:
            continue  # skip unreachable calm reference; case-study "normal" windows still exist
        ws = _slice_windows(series, symbol, event_id=None, crisis_start=None, crisis_end=None)
        for w in ws:
            windows_by_label.setdefault("normal", []).append(w)

    # Cap each EVENT class individually first -- COVID's ~33-day crisis span
    # naturally yields far fewer labeled windows than dot-com/GFC's multi-year
    # spans, so without a per-class cap those two would dominate cluster density.
    # Capping is done PER WINDOW LENGTH (stratified), not by flat random
    # subsampling of the pooled class: short lengths generate far more stride
    # positions than long ones, so flat random sampling would overwhelmingly
    # keep short windows -- but the dot-com/GFC declines were slow, grinding
    # bear markets where most SHORT sub-windows don't look dramatically
    # different from a calm period; it's the sparse LONG windows (spanning
    # most/all of the crisis) that carry the actually distinctive shape. This
    # keeps long-window representation from being crowded out.
    event_labels = [l for l in windows_by_label if l != "normal"]
    for label in event_labels:
        windows_by_label[label] = _stratified_cap(windows_by_label[label], EVENT_CLASS_CAP, rng)

    total_event_count = sum(len(windows_by_label[l]) for l in event_labels)
    normal_cap = max(int(total_event_count * NORMAL_CAP_RATIO), 30)
    if "normal" in windows_by_label:
        windows_by_label["normal"] = _stratified_cap(windows_by_label["normal"], normal_cap, rng)

    corpus = [w for label in windows_by_label for w in windows_by_label[label]]
    if len(corpus) < 20:
        raise RuntimeError("Reference corpus build produced too few windows to fit a matcher")
    return corpus


def _fit_hdbscan_with_fallback(X: np.ndarray, corpus_labels: list[str]) -> tuple[np.ndarray | None, str]:
    """
    Search a small grid of (min_cluster_size, min_samples) and pick the
    configuration that best separates the 3 named events into their own
    clean clusters -- NOT the one with the most clusters (a naive "more
    clusters = better" score just fragments the large "normal" pool into
    dozens of tiny near-duplicate clusters, which is worse for a report).

    Score rewards: (a) how many of {dotcom, gfc, covid} get at least one
    cluster where they're the purity>0.6 majority, weighted most heavily;
    (b) high size-weighted average purity; (c) low noise fraction; and
    lightly penalizes a large cluster count (fragmentation).
    """
    event_ids = {"dotcom", "gfc", "covid"}
    candidates = [(4, 3), (5, 3), (5, 5), (6, 4), (6, 5), (8, 5), (10, 5), (10, 8), (15, 10)]
    best = None
    for mcs, ms in candidates:
        if mcs > max(len(X) // 3, 2):
            continue
        try:
            cl = HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit_predict(X)
        except Exception:
            continue
        n_clusters = len(set(cl) - {-1})
        noise_frac = float((cl == -1).mean())
        if n_clusters < 2 or noise_frac >= 0.7:
            continue

        weighted_purity_sum, total_members, events_covered = 0.0, 0, set()
        for cid in set(cl) - {-1}:
            idxs = np.where(cl == cid)[0]
            member_labels = [corpus_labels[i] for i in idxs]
            majority_label, purity = _majority_and_purity(member_labels)
            weighted_purity_sum += purity * len(idxs)
            total_members += len(idxs)
            if majority_label in event_ids and purity > CLUSTER_PURITY_THRESHOLD:
                events_covered.add(majority_label)
        avg_purity = weighted_purity_sum / max(total_members, 1)

        score = len(events_covered) * 10 + avg_purity * 5 - noise_frac * 3 - 0.05 * n_clusters
        if best is None or score > best[0]:
            best = (score, cl)
    if best is not None:
        return best[1], "hdbscan_centroid"
    return None, "label_centroid_fallback"


def _cluster_stats(corpus: list[CorpusWindow], embeddings: np.ndarray, idxs: np.ndarray) -> dict:
    member_labels = [corpus[i].label for i in idxs]
    majority_label, purity = _majority_and_purity(member_labels)
    centroid = embeddings[idxs].mean(axis=0)
    dists = np.linalg.norm(embeddings[idxs] - centroid, axis=1)
    typical_dist = float(np.percentile(dists, 90)) if len(dists) > 1 else float(dists.max() if len(dists) else 0.0)
    typical_dist = max(typical_dist, 1e-3)  # floor so a tiny/singleton cluster isn't unmatchable
    # Restrict both the representative window and the "what happened next"
    # forward-return stats to members that actually CARRY the majority label
    # -- not the whole (possibly mixed) cluster. Otherwise a report claiming
    # "similar GFC-shaped windows were followed by +X%" could be silently
    # averaging in some off-label (e.g. dot-com-era) windows, and the
    # displayed "representative historical window" could show a misleading
    # date range for the label being reported.
    majority_mask = np.array(member_labels) == majority_label
    majority_local_idxs = np.where(majority_mask)[0]
    majority_global_idxs = idxs[majority_local_idxs]
    fwd_rets = [corpus[i].forward_return_21d for i in majority_global_idxs if corpus[i].forward_return_21d is not None]
    best_local = majority_local_idxs[np.argmin(dists[majority_local_idxs])]
    representative_idx = int(idxs[best_local])
    return {
        "majority_label": majority_label, "purity": purity, "centroid": centroid,
        "typical_dist": typical_dist,
        "forward_return_21d_mean": float(np.mean(fwd_rets)) if fwd_rets else None,
        "forward_return_21d_std": float(np.std(fwd_rets)) if fwd_rets else None,
        "forward_return_21d_n": len(fwd_rets),
        "representative_idx": representative_idx,
    }


def _local_purity(query_emb: np.ndarray, matcher: "Matcher", label: str, k: int = LOCAL_PURITY_K) -> float:
    """
    Fraction of the k nearest CORPUS WINDOWS to this exact query -- by raw
    embedding distance across the WHOLE corpus, not membership in whichever
    single HDBSCAN cluster the query happened to land nearest to -- that
    carry `label`.

    This is what actually gates/scales confidence (see match_window), in
    place of that cluster's own stored average purity. A cluster's average
    purity is a property of the WHOLE bucket and can be dragged down by
    contamination sitting anywhere in it; a query landing exactly on that
    cluster's cleanest, most prototypical point would then be capped at the
    bucket-wide average regardless of how unambiguous ITS OWN neighborhood
    actually is. Recomputing purity fresh from the query's own nearest
    neighbors fixes that -- confidence now reflects how clean the embedding
    space is right where this query landed.

    Votes are inverse-distance weighted rather than a flat per-neighbor
    count: a dot-com window and a covid window can both be genuine
    crash-then-recover shapes, so a handful of dot-com neighbors sitting
    just past the closest covid ones would otherwise dilute purity even
    though they're clearly farther away than the covid match itself.
    """
    dists = np.linalg.norm(matcher.embeddings - query_emb, axis=1)
    k = min(k, len(dists))
    nearest_idxs = np.argpartition(dists, k - 1)[:k]
    weights = 1.0 / (dists[nearest_idxs] + 1e-3)
    same_label = np.array([matcher.corpus[i].label == label for i in nearest_idxs])
    return float(weights[same_label].sum() / weights.sum())


def _label_scales(corpus: list[CorpusWindow], embeddings: np.ndarray) -> dict[str, float]:
    """
    Per-label reference scale: the p75, over that label's own corpus
    members, of each member's mean distance to its MATCH_K nearest
    same-label neighbours. I.e. "how far apart do genuine windows of this
    label sit from each other".

    An upper percentile rather than the median, because classes are built
    from overlapping slices of a few contiguous stretches and so contain
    many near-duplicate windows -- "normal" is 679 windows drawn from four
    calm periods, which crushes its median spacing to 0.10 against
    0.24-0.28 for the event classes purely as an artefact of duplication.
    Scoring against that median made textbook-calm windows read 0%
    confident. The p75 (0.16 vs 0.33-0.37) is far less sensitive to
    duplicate density, which suits what this term is actually for: an
    is-this-even-in-the-neighbourhood check, not a fine similarity gradient
    (the purity term supplies the discrimination).

    This replaces the old per-HDBSCAN-cluster p90-distance-to-centroid as
    the yardstick a query's distance is measured in. A centroid is the MEAN
    of a cluster, but HDBSCAN produces arbitrary-shaped density clusters,
    not spherical ones -- the mean of an elongated or multi-modal cluster
    can land in empty space where no member actually lives, so a query
    sitting exactly on top of real members could still be several
    "centroid radii" away and score near zero. Point-to-point spacing has
    no such failure mode.
    """
    labels = np.array([w.label for w in corpus])
    scales: dict[str, float] = {}
    for label in set(labels):
        member_emb = embeddings[labels == label]
        if len(member_emb) < 2:
            scales[label] = 1.0
            continue
        k = min(MATCH_K, len(member_emb) - 1)
        spacings = []
        for i in range(len(member_emb)):
            d = np.sort(np.linalg.norm(member_emb - member_emb[i], axis=1))[1:k + 1]
            spacings.append(float(d.mean()))
        scales[label] = max(float(np.percentile(spacings, 75)), 1e-3)
    return scales


def _score_label(query_emb: np.ndarray, matcher: "Matcher", label: str) -> dict:
    """
    Score how strongly this query resembles `label`, using the actual
    labelled corpus windows rather than any cluster summary -- so HDBSCAN
    noise points (~16% of the corpus, belonging to no cluster at all) still
    contribute, instead of being invisible to the score.

    match_dist is the mean distance to the MATCH_K nearest windows CARRYING
    that label; it's expressed in units of that label's own in-class
    spacing (_label_scales), then passed through a Gaussian kernel. That
    calibrates the way a reader expects: a query sitting well inside the
    spacing genuine members keep from each other reads ~0.9, one sitting at
    exactly the typical spacing reads ~0.6, and one 2-3x outside decays to
    ~0.1 and below.
    """
    dists = np.linalg.norm(matcher.embeddings - query_emb, axis=1)
    same_label_idxs = np.array([i for i, w in enumerate(matcher.corpus) if w.label == label])
    k = min(MATCH_K, len(same_label_idxs))
    nearest_same = same_label_idxs[np.argsort(dists[same_label_idxs])[:k]]

    match_dist = float(dists[nearest_same].mean())
    scale = matcher.label_scales[label]
    proximity = float(np.exp(-0.5 * (match_dist / scale) ** 2))
    purity = _local_purity(query_emb, matcher, label)

    fwd = [matcher.corpus[i].forward_return_21d for i in nearest_same
           if matcher.corpus[i].forward_return_21d is not None]
    return {
        "label": label,
        "confidence": proximity * purity,
        "purity": purity,
        "match_dist": match_dist,
        "typical_dist": scale,
        "nearest_idx": int(nearest_same[0]),
        "neighbour_idxs": nearest_same,
        "forward_return_21d_mean": float(np.mean(fwd)) if fwd else None,
        "forward_return_21d_std": float(np.std(fwd)) if fwd else None,
        "forward_return_21d_n": len(fwd),
    }


def _build_cluster_meta(corpus: list[CorpusWindow], embeddings: np.ndarray,
                         cluster_labels: np.ndarray | None, fit_method: str) -> dict:
    if fit_method == "hdbscan_centroid":
        return {
            int(cid): _cluster_stats(corpus, embeddings, np.where(cluster_labels == cid)[0])
            for cid in sorted(set(cluster_labels) - {-1})
        }
    all_labels = np.array([w.label for w in corpus])
    return {
        i: _cluster_stats(corpus, embeddings, np.where(all_labels == label)[0])
        for i, label in enumerate(sorted(set(all_labels)))
    }


def get_matcher(force_rebuild: bool = False) -> Matcher:
    global _MATCHER
    if _MATCHER is not None and not force_rebuild:
        return _MATCHER
    with _MATCHER_LOCK:
        if _MATCHER is not None and not force_rebuild:
            return _MATCHER
        corpus = _build_corpus()
        channels_stack = np.stack([w.resampled for w in corpus])
        mean, std = embedding.fit_scaler(channels_stack)
        encoder = embedding.train_encoder(channels_stack, mean, std)
        embeddings = embedding.embed(encoder, channels_stack, mean, std)
        cluster_labels, fit_method = _fit_hdbscan_with_fallback(embeddings, [w.label for w in corpus])
        cluster_meta = _build_cluster_meta(corpus, embeddings, cluster_labels, fit_method)
        _MATCHER = Matcher(encoder=encoder, scaler_mean=mean, scaler_std=std, corpus=corpus,
                            embeddings=embeddings, cluster_meta=cluster_meta, fit_method=fit_method,
                            label_scales=_label_scales(corpus, embeddings))
        return _MATCHER


def _generate_insights(payload: dict) -> list[str]:
    m, qs, ref, base = payload["match"], payload["query_stats"], payload["reference"], payload["baseline"]
    n_days = payload["window_trading_days"]
    bullets = []

    if m["event_id"] and m["matched"]:
        bullets.append(
            f"This {n_days}-trading-day window's shape most closely resembles the {m['event_label']} "
            f"({m['confidence'] * 100:.0f}% confidence, based on how close it sits to the nearest "
            f"historical {m['event_label']} windows in the reference corpus, and how unambiguously "
            f"that neighbourhood belongs to that event)."
        )
    elif m["event_id"]:
        # Targeted mode (a specific event was asked about) but the gate
        # didn't clear -- still report the confidence number rather than
        # the generic "reads as normal" line, which would misstate what was
        # actually being measured.
        bullets.append(
            f"This {n_days}-trading-day window shows only a weak resemblance to the {m['event_label']} "
            f"({m['confidence'] * 100:.0f}% confidence against the nearest historical windows of that "
            f"event) — not a close match."
        )
    else:
        bullets.append(
            f"This {n_days}-trading-day window does not closely resemble any of the three historical "
            f"crises tracked (dot-com bust, 2008 GFC, COVID crash) — it reads as a normal / typical "
            f"market period, with {m['confidence'] * 100:.0f}% confidence in that 'normal' reading "
            f"(the percentage below is confidence in this classification, not in a crisis match)."
        )

    bullets.append(
        f"Over this window: {qs['annualized_return'] * 100:+.1f}% annualized return, "
        f"{qs['annualized_volatility'] * 100:.1f}% annualized volatility, and a maximum drawdown of "
        f"{qs['max_drawdown'] * 100:.1f}%."
    )

    if qs["regime_mix"]:
        rm = qs["regime_mix"]
        bullets.append(
            f"The day-by-day HMM regime model classified this window as {rm['bull'] * 100:.0f}% Bull, "
            f"{rm['bear'] * 100:.0f}% Bear, and {rm['crisis'] * 100:.0f}% Crisis days."
        )

    fwd_mean = base["forward_return_21d_mean"]
    if fwd_mean is not None:
        direction = "gained" if fwd_mean >= 0 else "lost"
        bullets.append(
            f"{base['label']} were followed by a next-21-trading-day return averaging "
            f"{fwd_mean * 100:+.1f}% (the market typically {direction} ground over the following month; "
            f"std {base['forward_return_21d_std'] * 100:.1f}%, n={base['n_samples']} similar historical windows)."
        )

    if m["event_id"]:
        bullets.append(
            f"A representative historical {ref['event_label']} window ({ref['window_start']} to "
            f"{ref['window_end']}, {ref['symbol']}) is plotted alongside this period's own trend shape "
            f"in the chart above for a direct visual comparison."
        )
        # The chart only plots the return-path channel; the similarity score
        # also weighs volatility/drawdown intensity, which can look wildly
        # different even when the two return curves track closely -- flag it
        # explicitly whenever that gap looks large enough to explain a
        # surprising confidence number, rather than leaving a "the shapes
        # looked alike but the score was low" mismatch unexplained.
        q_dd, r_dd = abs(qs["max_drawdown"]), abs(ref["max_drawdown"])
        q_vol, r_vol = qs["peak_rolling_volatility"], ref["peak_rolling_volatility"]
        dd_ratio = max(q_dd, r_dd) / max(min(q_dd, r_dd), 1e-4)
        vol_ratio = max(q_vol, r_vol) / max(min(q_vol, r_vol), 1e-4)
        if not m["matched"] and (dd_ratio > 1.8 or vol_ratio > 1.8):
            bullets.append(
                f"Even where the return curves above track similarly, the similarity score also weighs "
                f"drawdown depth and volatility intensity: this window saw a {qs['max_drawdown'] * 100:.1f}% "
                f"drawdown with volatility peaking around {q_vol * 100:.0f}% (annualized), versus the "
                f"representative {m['event_label']} window's {ref['max_drawdown'] * 100:.1f}% drawdown and "
                f"~{r_vol * 100:.0f}% peak volatility — that gap is a large part of why the confidence stayed low."
            )

    return bullets


def match_window(symbol: str, start: str, end: str, target_event: str | None = None) -> dict:
    """
    If `target_event` is None, finds whichever reference cluster (any of the
    3 named events, or "normal") the query window is nearest to -- the
    "what does this look like, if anything" mode.

    If `target_event` is one of EVENT_LABELS' event ids ("dotcom"/"gfc"/
    "covid"), the search is restricted to THAT event's own clusters --
    "how closely does this resemble specifically the {target_event}, on its
    own terms" -- so the confidence returned always answers the question the
    caller actually asked, rather than silently reporting on whichever event
    happened to be globally closest (which could be a different one, or
    "normal", even when the caller cares about one specific event).
    """
    hist = md.get_history(symbol, start=start, end=end)  # raises MarketDataError upstream
    prices = hist["adjclose"].dropna()
    if len(prices) < MIN_QUERY_DAYS:
        raise ValueError(
            f"Selected window too short for pattern comparison — need at least "
            f"{MIN_QUERY_DAYS} trading days (got {len(prices)})"
        )
    if target_event is not None and target_event not in ("dotcom", "gfc", "covid"):
        raise ValueError(f"Unknown target_event '{target_event}'")

    matcher = get_matcher()

    channels = embedding.build_channels(prices)
    resampled = embedding.resample_to_fixed(channels)
    query_emb = embedding.embed(matcher.encoder, resampled[None], matcher.scaler_mean, matcher.scaler_std)[0]

    if target_event is not None:
        if not any(w.label == target_event for w in matcher.corpus):
            raise ValueError(f"Reference corpus has no '{target_event}' windows to compare against")
        best = _score_label(query_emb, matcher, target_event)
    else:
        # Rank the candidate labels by the confidence each one actually
        # yields, not by raw distance. Each label has its own in-class
        # spacing, so raw distances across labels aren't comparable
        # quantities -- picking by raw distance systematically favours
        # whichever label happens to be densest ("normal", which outnumbers
        # every event class here).
        scored = [_score_label(query_emb, matcher, lab) for lab in sorted(EVENT_LABELS)
                  if any(w.label == lab for w in matcher.corpus)]
        best = max(scored, key=lambda s: s["confidence"])

    best_dist, typical_dist = best["match_dist"], best["typical_dist"]
    local_purity, confidence = best["purity"], best["confidence"]
    is_event = target_event is not None or (
        best["label"] != "normal" and local_purity > CLUSTER_PURITY_THRESHOLD
    )
    matched = (best_dist <= typical_dist * DISTANCE_SLACK) and (confidence >= CONFIDENCE_THRESHOLD) and is_event

    try:
        regime_result = detect_regimes(prices)
        vc = pd.Series(regime_result.state_key).value_counts(normalize=True)
        regime_mix = {k: round(float(vc.get(k, 0.0)), 4) for k in ("bull", "bear", "crisis")}
    except ValueError:
        regime_mix = None  # window too short for the HMM's own >=30-day requirement

    daily_ret = prices.pct_change().dropna()
    years = max(len(prices) / 252, 1e-6)
    ann_return = float((prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1) if prices.iloc[0] > 0 else 0.0
    ann_vol = float(daily_ret.std() * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
    max_dd = float((prices / prices.cummax() - 1).min())

    query_stats = {
        "annualized_return": round(ann_return, 4), "annualized_volatility": round(ann_vol, 4),
        "max_drawdown": round(max_dd, 4), "regime_mix": regime_mix,
        "cumulative_return_path": [round(float(v), 4) for v in resampled[:, 0]],
        # Peak of the short rolling-vol channel (not the whole-window annualized
        # vol above) -- this is the same per-point intensity the embedding
        # actually sees. The chart only plots cumulative_return_path (channel
        # 0), so two windows can look alike there while differing sharply on
        # this -- see the "shape similarity weighs..." insight below.
        "peak_rolling_volatility": round(float(resampled[:, 1].max()), 4),
    }

    if target_event is not None:
        # Targeted mode always reports on the event that was asked about --
        # `matched` still flags whether it's a genuinely close resemblance,
        # but the confidence number and reference window are always the
        # target event's, even at low confidence, rather than silently
        # swapping to "normal" the caller never asked about.
        event_id, ref = target_event, best
    elif matched:
        event_id, ref = best["label"], best
    else:
        # Nothing convincing: show what an ordinary calm window looks like
        # instead, scored the same way so the reference is the query's own
        # nearest normal windows rather than an unrelated cluster.
        event_id = None
        ref = _score_label(query_emb, matcher, "normal") if best["label"] != "normal" else best
        # Report the NORMAL score too, not the rejected event's. `best` here
        # may be an event label that lost the purity gate, and surfacing its
        # number under a "reads as normal / typical" verdict states one
        # label's confidence while naming another -- the reader has no way to
        # tell the 22% describes dot-com rather than the normal call.
        local_purity, confidence = ref["purity"], ref["confidence"]
        best_dist, typical_dist = ref["match_dist"], ref["typical_dist"]

    match_payload = {
        "matched": matched, "event_id": event_id,
        "event_label": EVENT_LABELS[event_id] if event_id else None,
        # Local, query-specific purity (see _local_purity) -- what actually
        # scales `confidence` above -- not a cluster's static average, which
        # the local figure can legitimately run well above or below depending
        # on exactly where the query landed.
        "confidence": round(confidence, 4), "cluster_purity": round(local_purity, 4),
        "match_method": "knn_shape_similarity",
        "match_distance": round(best_dist, 4), "typical_distance": round(typical_dist, 4),
    }

    rep_window = matcher.corpus[ref["nearest_idx"]]
    reference_payload = {
        "event_id": ref["label"] if ref["label"] != "normal" else None,
        "event_label": EVENT_LABELS[ref["label"]],
        "window_start": rep_window.start, "window_end": rep_window.end, "symbol": rep_window.symbol,
        "cumulative_return_path": [round(float(v), 4) for v in rep_window.resampled[:, 0]],
        "max_drawdown": round(float(rep_window.resampled[:, 2].min()), 4),
        "peak_rolling_volatility": round(float(rep_window.resampled[:, 1].max()), 4),
        "cluster_forward_return_21d": {
            "mean": round(ref["forward_return_21d_mean"], 4) if ref["forward_return_21d_mean"] is not None else None,
            "std": round(ref["forward_return_21d_std"], 4) if ref["forward_return_21d_std"] is not None else None,
            "n": ref["forward_return_21d_n"],
        },
    }

    baseline_payload = {
        "label": (f"Historically similar '{EVENT_LABELS[ref['label']]}'-shaped windows"
                  if (matched or target_event is not None) else "Typical 'normal' market windows"),
        "forward_return_21d_mean": reference_payload["cluster_forward_return_21d"]["mean"],
        "forward_return_21d_std": reference_payload["cluster_forward_return_21d"]["std"],
        "n_samples": reference_payload["cluster_forward_return_21d"]["n"],
    }

    payload = {
        "symbol": symbol, "start": start, "end": end, "window_trading_days": len(prices),
        "match": match_payload, "query_stats": query_stats, "reference": reference_payload,
        "baseline": baseline_payload,
    }
    payload["insights"] = _generate_insights(payload)
    payload["disclaimer"] = (
        "Illustrative shape-similarity analysis against 3 historical case studies plus a calm-period "
        "reference corpus, for a student demo project — not investment advice, not predictive."
    )
    return payload
