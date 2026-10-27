"""Chapters 5 to 9 and Appendix A of the project report."""
import ast
import textwrap

import pandas as pd

pct = lambda x, d=1: f"{x * 100:.{d}f}%"
CASES = ["dotcom_deployed", "dotcom_corrected", "gfc", "covid"]
CNAME = {"dotcom_deployed": "Dot-com (as deployed)", "dotcom_corrected": "Dot-com (corrected)", "gfc": "2008 GFC",
         "covid": "COVID-19"}
SNAME = {"equal_weight": "Equal-Weight", "mean_variance": "Mean-Variance", "adaptive": "Adaptive (ES)"}


def tidy(s: str) -> str:
    """Remove dash characters from application messages quoted in the report."""
    return s.replace(" — ", ", ").replace("—", ", ").replace("–", "-")


def extract_functions(path, names):
    src = open(path).read()
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            out.append((node.lineno, ast.get_source_segment(src, node)))
    out.sort()
    return [seg for _, seg in out]


def wrap_code(seg, width=88):
    lines = []
    seg = seg.replace("\u2014", "-").replace("\u2013", "-")
    for ln in seg.splitlines():
        if len(ln) <= width:
            lines.append(ln)
        else:
            indent = len(ln) - len(ln.lstrip())
            parts = textwrap.wrap(ln.strip(), width - indent - 4, break_long_words=False)
            lines.append(" " * indent + parts[0])
            lines.extend(" " * (indent + 4) + p for p in parts[1:])
    return lines


def write(R, D):
    FIGD, BT, M, ABL, TESTS, ROOT = D["FIG"], D["BT"], D["M"], D["ABL"], D["TESTS"], D["ROOT"]
    cv, lo, q = M["cv"], M["loeo"], M["queries"]
    TR, UN = cv["Transformer (trained)"], cv["Transformer (untrained)"]
    abl = D["abl"]
    ncorp = sum(M["corpus"].values())

    # ======================================================================== CHAPTER 5
    R.chapter(5, "Methodology and Testing")
    R.p("This chapter describes the modules of the system in the order data flows through them, and then the tests used "
        "to check the software and the models. Results are reported in Chapter 7.")
    R.h2("5.1 Module Description")
    R.h3("5.1.1 Data Acquisition and Preprocessing")
    R.p("All results use real daily adjusted close prices from the Yahoo Finance chart interface {cite:yahoo}. The client "
        "in market_data.py calls the endpoint through curl with explicit start and end timestamps. Yahoo's default \"max\" "
        "range quietly returns roughly monthly bars for long histories, so explicit timestamps are needed to get daily data. "
        "Failed requests are retried with back-off, and every response is cached on disk for 15 minutes. If the feed cannot be "
        "reached, the three case studies can run on a synthetic series calibrated to the size of each crisis, and the response "
        "carries a flag so the dashboard can say so. No result in this report used the synthetic fallback.")
    R.p("{tab:t_cases} defines the three crises and the four backtest settings. The dot-com setting appears twice. The "
        "version labelled \"as deployed\" uses gold futures (GC=F), whose Yahoo history only begins in August 2000, after the "
        "crisis had started. The \"corrected\" version replaces them with the Treasury fund VUSTX, which has a long history, "
        "so that a real pre-crisis training period exists (Section 7.3).")
    rows = []
    for k in CASES:
        c = BT["cases"][k]
        rows.append([CNAME[k], ", ".join(c["assets"]), f"{c['date_range']['start']} to {c['date_range']['end']}", c["n_days"],
                     f"{c['crisis_window'][0]} to {c['crisis_window'][1]}", f"{c['n_train_days']} / {c['n_test_days']}"])
    R.table("t_cases", ["Setting", "Assets", "Data window", "Trading days", "Crisis window", "Train / test days"], rows,
            "Backtest settings and crisis definitions.", widths=[1.1, 1.1, 1.45, 0.65, 1.45, 0.65], font=8.5,
            align=["l", "l", "c", "c", "c", "c"])
    rows = []
    for k in CASES:
        for a, s in BT["dataset"][k].items():
            rows.append([CNAME[k], a, f"{s['start']} to {s['end']}", s["n"], pct(s["ann_return"]), pct(s["ann_vol"]), pct(s["max_drawdown"])])
    R.table("t_assets", ["Setting", "Asset", "Period", "N", "Ann. return", "Ann. volatility", "Max drawdown"], rows,
            "Descriptive statistics of each asset over each backtest window (252 trading days per year).",
            widths=[1.2, 0.7, 1.55, 0.5, 0.8, 0.9, 0.85], font=8.5, align=["l", "l", "c", "c", "c", "c", "c"])
    R.p(f"For the pattern comparison we built a reference corpus of {ncorp} labelled windows. They come from the three case "
        f"studies (QQQ for the dot-com bust, SPY for the other two) and from four calm periods: SPY in 2003 to 2006, 2012 "
        f"to 2015 and 2016 to 2019, and QQQ in 2012 to 2015. The corpus holds {M['corpus']['normal']} normal, "
        f"{M['corpus']['dotcom']} dot-com, {M['corpus']['gfc']} GFC and {M['corpus']['covid']} COVID-19 windows.")
    R.p("Prices from different assets are aligned on shared trading dates, and daily simple returns are computed for the "
        "backtests. For regime detection, each day is described by three features: the log return, the 20-day annualized "
        "volatility and the drawdown from the 60-day rolling peak.")
    R.equation(r"r_t=\ln\frac{P_t}{P_{t-1}}")
    R.equation(r"\sigma_t=\sqrt{252}\,\mathrm{std}\,(r_{t-19},\ldots,r_t)")
    R.equation(r"d_t=\frac{P_t}{\max_{s\in[t-59,\,t]}P_s}-1")
    R.p("The features are standardized before the model is fitted.")

    R.h3("5.1.2 Regime Detection with a Hidden Markov Model")
    R.p("A three-state Gaussian hidden Markov model with diagonal covariance {cite:rabiner89} is fitted by "
        "expectation-maximization (300 iterations, seed 7) on the standardized features. Each state has its own mean and "
        "variance for the three features, and a transition matrix describes how likely the market is to move between states.")
    R.equation(r"z_t\mid z_{t-1}=i\sim\mathrm{Cat}(A_{i\cdot}),\qquad \mathbf{x}_t\mid z_t=j\sim\mathcal{N}\left(\mu_j,\mathrm{diag}(\sigma_j^{2})\right)")
    R.p("State numbers are arbitrary, so we name the states afterwards from their statistics. The state with the highest "
        "average volatility is Crisis. Of the other two, the one with the higher average return is Bull and the other is Bear. "
        "The model never sees crash dates. Its posterior probabilities are used by the dashboard and as part of the state "
        "of the adaptive allocator.")

    R.h3("5.1.3 Length-Invariant Shape and Transformer Encoder")
    R.p("The comparison has to treat a 33-day crash and a 600-day bear market on equal terms. For a window of n trading days "
        "we compute three channels from the window's own start: cumulative return, a 5-day rolling annualized volatility, and "
        "the running drawdown. Each channel is linearly interpolated onto 40 equally spaced points of a normalized time axis, "
        "so every window becomes a 40 by 3 array.")
    R.equation(r"c_\tau=\frac{P_\tau}{P_1}-1")
    R.equation(r"v_\tau=\sqrt{252}\,\mathrm{std}(\rho_{\tau-4:\tau})")
    R.equation(r"g_\tau=\frac{P_\tau}{\max_{s\leq\tau}P_s}-1,\qquad \tilde{X}\in\mathbb{R}^{40\times 3}")
    R.p("Here ρ stands for the daily returns inside the window. The channels are standardized with means and standard "
        "deviations computed over the corpus.")
    R.p("A small Transformer encoder {cite:vaswani17} maps the 40 by 3 array to a 16-dimensional embedding ({fig:transformer}). "
        "A linear layer lifts each time step to 32 dimensions and a learned positional embedding is added. Two encoder layers "
        "with four attention heads and a 64-unit feed-forward block (GELU activation {cite:hendrycks16}, layer "
        "normalization {cite:ba16}) follow. The 40 outputs are averaged and a last linear layer produces the 16 values. Self-attention "
        "is the scaled dot product of Vaswani et al.:")
    R.equation(r"\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V")
    R.p("The encoder is trained without labels as an autoencoder. A decoder of the same size rebuilds the 40 by 3 array from "
        "the embedding, and training minimizes the squared reconstruction error with Adam {cite:kingma15} (learning rate "
        "0.002, 120 full-batch epochs) in PyTorch {cite:paszke19}. The decoder is thrown away after training.")
    R.equation(r"\mathcal{L}=\frac{1}{N}\sum_{i=1}^{N}\left\|\tilde{X}_i-g_\phi\left(f_\theta(\tilde{X}_i)\right)\right\|_F^{2}")
    R.figure("transformer", FIGD / "fig_transformer.png", "Transformer shape encoder and the decoder used only for training.", width=6.0)

    R.h3("5.1.4 Reference Corpus and HDBSCAN")
    R.p("Overlapping windows are cut from each source series at 21 lengths between 10 and 600 trading days, with a stride "
        "of max(3, length/5) days. A window gets an event label if the crisis covers at least half of the window, or if the "
        "window contains at least 80 percent of the crisis and is no longer than four times its length. The second rule "
        "exists because the COVID-19 crash lasted only 33 trading days, so most windows that contain it are longer than the "
        "crash itself. Windows that overlap a crisis by 5 to 50 percent are dropped as ambiguous, and the rest are labelled "
        "normal. Each event class is capped at 120 windows and the normal class at 2.5 times the event total. The cap is "
        "stratified by window length, so a large number of short windows does not push out the rarer long ones.")
    R.p("HDBSCAN {cite:campello13,mcinnes17} is run on the embeddings over a grid of nine settings of minimum cluster size "
        "and minimum samples. We choose the setting that gives each event its own cluster with a majority of at least 60 "
        "percent, with high purity and little noise, and that avoids splitting into many small clusters. On the deployed "
        f"corpus it finds {M['matcher']['n_clusters']} clusters. We use the clustering to inspect the structure of the "
        "corpus. The final score does not rely on cluster centers, because the center of an irregular density cluster can "
        "lie in empty space, and about 16 percent of the corpus is classed as noise that no center describes.")

    R.h3("5.1.5 Similarity Scoring and the Comparison Report")
    R.p("For a query embedding z and a label ℓ, let N_k(z) be the k = 8 corpus windows nearest to z that carry label ℓ, and let "
        "s_ℓ be the typical in-class spacing, the 75th percentile over the label's windows of the mean distance to their "
        "eight nearest same-label neighbors. The distance score m_ℓ is turned into a similarity by a Gaussian kernel, and "
        "a purity term p_ℓ measures how many of the 15 nearest windows overall carry the same label, weighted by inverse distance.")
    R.equation(r"m_\ell(z)=\frac{1}{k}\sum_{j\in\mathcal{N}_k^{\ell}(z)}\left\|z-z_j\right\|_2")
    R.equation(r"\pi_\ell(z)=\exp\left[-\frac{1}{2}\left(\frac{m_\ell(z)}{s_\ell}\right)^{2}\right]")
    R.equation(r"p_\ell(z)=\frac{\sum_{j\in\mathcal{N}_{15}(z)}w_j\,\mathbb{1}[y_j=\ell]}{\sum_{j\in\mathcal{N}_{15}(z)}w_j}")
    R.equation(r"w_j=\frac{1}{\left\|z-z_j\right\|_2+10^{-3}}")
    R.equation(r"C_\ell(z)=\pi_\ell(z)\,p_\ell(z)")
    R.p("In targeted mode the user picks an event and the confidence C_ℓ is reported for that event. In auto mode the label "
        "with the highest C_ℓ wins. A match requires m_ℓ ≤ 1.4 s_ℓ and C_ℓ ≥ 0.20, and in auto mode the winner must not be "
        "the normal class and must have p_ℓ above 0.6. If these conditions fail, the report says that no crisis resembles the "
        "period and gives the normal-regime confidence instead. Every report also contains the nearest same-label real "
        "window for the overlay, the annualized return, volatility and maximum drawdown of the period, the share of "
        "bull, bear and crisis days from the HMM, and the average 21-day forward return of the nearest same-label windows "
        "as a historical baseline. {fig:pipeline} gives the whole pipeline.")
    R.figure("pipeline", FIGD / "fig_pipeline_matcher.png", "Historical-pattern comparison pipeline.", width=6.0)

    R.h3("5.1.6 Allocation Strategies")
    R.p("Three strategies are compared on the same rebalance dates (every 21 trading days) and with the same proportional "
        "cost of 5 basis points on turnover.")
    R.bullets([
        "**Equal-Weight** holds 1/N of each asset.",
        "**Mean-Variance** maximizes the Sharpe ratio of a long-only, fully invested portfolio using the previous 63 trading "
        "days, solved with sequential least squares programming. If the solver fails it falls back to equal weights.",
        "**Adaptive** is a small neural policy. Its input is the state s_k: the three regime probabilities, and the 21-day "
        "momentum and 21-day annualized volatility of every asset. A hidden layer of 16 tanh units and a softmax output give "
        "long-only weights.",
    ])
    R.equation(r"\mathbf{w}_k=\mathrm{softmax}\left(W_2\tanh\left(W_1\mathbf{s}_k+\mathbf{b}_1\right)+\mathbf{b}_2\right)")
    R.p("The policy is trained with evolution strategies {cite:salimans17}. In each of 40 generations, 32 antithetic "
        "Gaussian perturbations of the parameter vector (σ = 0.1) are scored by simulating the policy through the training "
        "segment. The rewards are rank-normalized and the parameters move along the reward-weighted perturbation direction "
        "with a step of 0.05. The reward subtracts a drawdown penalty and a turnover penalty from the Sharpe ratio, which is "
        "how transaction costs enter the objective.")
    R.equation(r"R(\theta)=\mathrm{SR}-2\,\mathrm{MDD}-0.5\,\overline{\mathrm{TO}}")
    R.equation(r"\theta\leftarrow\theta+\frac{\alpha}{N\sigma}\sum_{i=1}^{N}\hat{F}_i\,\varepsilon_i")
    R.figure("protocol", FIGD / "fig_backtest_protocol.png", "Walk-forward protocol: training on the pre-crisis segment, evaluation through the crisis.", width=6.0)

    R.h3("5.1.7 Backtest Engine and Evaluation Metrics")
    R.p("The engine splits each setting at the crisis start. The adaptive policy and the 63-day look-back of the "
        "mean-variance strategy only ever use data before the date being decided. The test segment runs from the crisis start "
        "to the end of the data window. With daily portfolio returns r and portfolio value V, the metrics are:")
    R.equation(r"\mathrm{SR}=\sqrt{252}\,\frac{\overline{r-r_f/252}}{\mathrm{std}(r)}")
    R.equation(r"\mathrm{MDD}=\min_t\left(\frac{V_t}{\max_{s\leq t}V_s}-1\right)")
    R.equation(r"\mathrm{Calmar}=\frac{\mathrm{AnnRet}}{|\mathrm{MDD}|}")
    R.equation(r"\mathrm{TO}_k=\left\|\mathbf{w}_k-\mathbf{w}_{k-1}\right\|_1")
    R.equation(r"V\leftarrow V\left(1-c\,\mathrm{TO}_k\right),\qquad c=5\times10^{-4}")
    R.equation(r"\Delta=\mathrm{SR}_{\mathrm{adaptive}}-\mathrm{SR}_{\mathrm{baseline}}")
    R.p("The risk-free rate is 2 percent a year. Total return, annualized return, annualized volatility and mean turnover are "
        "also reported. For the matcher, the event classes are much smaller than the normal class, so we report balanced "
        "accuracy (the mean recall over the four classes, with a chance level of 0.25) and macro-F1, together with the "
        "crisis recall (the share of event windows predicted as some event) and the false-alarm rate (the share of normal "
        "windows predicted as an event). Plain accuracy is given only for completeness, since always answering \"normal\" "
        "already scores 70 percent.")

    R.h3("5.1.8 Dashboard")
    R.p("The React interface has three tabs. Portfolio Backtest lets the user choose a case study, runs the backtest and "
        "draws the value curves, the metrics table and the allocation chart. Regime Detection draws the price with colored "
        "regime bands for any symbol and range, and below it the Historical Pattern Comparison report with its overlay "
        "chart, key figures and written insights. About describes the method and the stack. A ticker ribbon at the top "
        "polls the live quotes. The charts use Recharts, and a request counter in the regime explorer ignores answers to "
        "outdated requests. {tab:t_hyper} collects the hyper-parameters of all modules.")
    R.table("t_hyper", ["Component", "Setting"], [
        ["HMM", "3 states, diagonal covariance, 300 EM iterations, seed 7; features: log return, 20-day volatility, 60-day drawdown"],
        ["Shape", "40 points by 3 channels; windows of 10 to 600 days (21 lengths); stride max(3, length/5)"],
        ["Transformer", "d_model 32, 4 heads, 2 layers, feed-forward 64 (GELU), learned positions, mean pooling, 16-d output; Adam, learning rate 0.002, 120 epochs"],
        ["Corpus", "Event cap 120 per class (length-stratified); normal cap 2.5 times the events; 50 percent overlap or 80 percent coverage rule"],
        ["HDBSCAN", "9-point grid on minimum cluster size and minimum samples; the score favors one majority cluster per event, high purity, low noise"],
        ["Scoring", "k = 8 same-label neighbors; Gaussian kernel on spacing s; purity over 15 nearest; match if m ≤ 1.4 s and C ≥ 0.20"],
        ["Backtest", "Rebalance every 21 trading days; cost 5 bps on turnover; risk-free rate 2 percent; 252 trading days a year"],
        ["Mean-Variance", "63-day window, long-only maximum Sharpe (SLSQP), equal-weight fallback"],
        ["ES policy", "16 tanh hidden units; 32 antithetic perturbations by 40 generations; σ 0.1; step 0.05; reward SR - 2 MDD - 0.5 turnover"],
    ], "Hyper-parameters used in the experiments.", widths=[1.2, 5.2], font=9.5, align=["l", "l"])

    # ------------------------------------------------------------------ testing
    R.h2("5.2 Testing")
    R.h3("5.2.1 Approach and Test Environment")
    R.p("The project has no earlier automated test suite, so for this report we wrote a test script "
        "(paper/scripts/run_tests.py) with 21 test cases and ran it on the development machine described in Section 3.3. "
        "The API cases call the real routes through FastAPI's test client with live Yahoo data, so they also check the data "
        "path. The unit cases check numerical properties of single modules. Every case was run once and the observed result "
        "was recorded; all 21 passed.")
    obs_fix = {
        "TC08": "HTTP 400; message says the window is too short and gives the count (7 trading days)",
        "TC11": "HTTP 502; message says Yahoo returned no data for that symbol",
        "TC13": "HTTP 200; response holds strategies, metrics for the test and full periods, regimes and the split date",
        "TC01": "HTTP 200; body holds a status field",
        "TC02": "HTTP 200; ids dotcom, gfc and covid",
        "TC03": "HTTP 200; assets SPY, QQQ, GLD, TLT",
        "TC12": "HTTP 400; message names the unknown case study",
        "TC16": "(40, 3) for windows of 33, 120 and 600 days",
        "TC17": "equal-weight sum 1.0000; mean-variance sum 1.0000 with minimum weight 0.0000",
        "TC18": "softmax weights sum to 1.0000 with minimum 0.016; flatten and rebuild give identical output",
        "TC19": "mean volatility: crisis 0.543, bear 0.220, bull 0.131",
        "TC20": "embeddings of shape (965, 16); 679 normal, 105 dot-com, 101 GFC and 80 COVID-19 windows; 3 HDBSCAN clusters",
        "TC21": "two identical queries gave confidence 0.5912 both times",
        "TC05": "HTTP 200; regime bear, probabilities sum to 1.000",
        "TC06": "HTTP 200; 988 daily labels, states bear, bull and crisis present",
        "TC09": "HTTP 200; matched, event covid, confidence 0.59",
        "TC10": "HTTP 200; no crisis matched, normal regime",
        "TC14": "108 rebalance rows checked; largest deviation of a weight sum from 1 is 1.0e-04 (rounding in the output)",
    }
    res = {r["id"]: r for r in TESTS["results"]}

    def trow(tid):
        r = res[tid]
        return [tid, r["title"], r["expected"], tidy(obs_fix.get(tid, r["observed"])), r["status"]]
    api_ids = [f"TC{n:02d}" for n in range(1, 14)]
    R.h3("5.2.2 Functional (API) Tests")
    R.table("t_apitests", ["ID", "Test case", "Expected result", "Observed result", "Result"], [trow(i) for i in api_ids],
            "Functional tests of the REST API.", widths=[0.5, 1.7, 1.45, 2.2, 0.55], font=8.5, align=["c", "l", "l", "l", "c"])
    R.h3("5.2.3 Unit Tests")
    unit_ids = [f"TC{n:02d}" for n in range(14, 22)]
    R.table("t_unittests", ["ID", "Test case", "Expected result", "Observed result", "Result"], [trow(i) for i in unit_ids],
            "Unit tests of the analytics modules.", widths=[0.5, 1.7, 1.45, 2.2, 0.55], font=8.5, align=["c", "l", "l", "l", "c"])
    R.p("TC15 compares the final value of the Mean-Variance strategy in the 2008 setting at 0 and at 50 basis points of cost. "
        "TC19 uses the SPY history from 2007 to 2010 and checks that the state named Crisis really is the highest-volatility "
        "state, which holds by construction but guards against a regression in the relabelling code.")
    R.h3("5.2.4 Model Validation Protocol")
    R.p("The software tests say nothing about whether the models are any good, so we designed four further experiments. "
        "Their results are in Chapter 7.")
    R.bullets([
        "**Blocked, purged cross-validation of the matcher.** Within each label the windows are sorted by start date and cut "
        "into five contiguous blocks. Any training window that overlaps in calendar time with a test window is removed, and the "
        "encoder is retrained on the training windows only. Ordinary shuffled cross-validation would be misleading here, "
        "because neighboring windows share most of their days.",
        "**Leave-one-event-out.** Train on two crises plus normal windows, purge anything overlapping the held-out crisis, "
        "and check whether the unseen crisis is flagged as any crisis class rather than normal.",
        "**Embedding ablation.** The same 8-nearest-neighbor classifier is run on the trained Transformer, an untrained "
        "random Transformer, the raw 120-value shape, a 16-component PCA of it, and seven hand-made summary statistics. If "
        "the trained Transformer does not beat the others, it adds nothing.",
        "**Allocator ablation and bootstrap.** Six seeds, four settings and three variants (full, no regime information, "
        "untrained policy), plus a paired stationary bootstrap {cite:politis94} (5,000 resamples, mean block length 10 days) "
        "of the Sharpe ratio difference between the adaptive policy and each baseline.",
    ])
    R.h3("5.2.5 Defects Found and Fixed")
    R.table("t_defects", ["Issue", "Symptom", "Cause", "Fix"], [
        ["Data rate limiting", "Early requests to Yahoo returned HTTP 429 in bursts", "Default client and repeated requests for the same history", "Direct curl client with a browser-like user agent, retries with back-off, 15-minute disk cache, labelled synthetic fallback"],
        ["Coarse history", "Long histories came back as roughly monthly bars", "Yahoo's \"max\" range coarsens the interval", "Explicit start and end timestamps in every request"],
        ["Stale chart", "The regime chart sometimes showed the previous selection", "Older responses arrived after newer ones and overwrote them", "Request counters in the regime explorer ignore outdated answers; load functions take explicit arguments"],
        ["Shape mismatch", "Dot-com queries were compared with a corpus built from a different symbol", "Corpus used SPY while the shape of the event is clearest in QQQ", "A shape_symbol field in the case-study configuration, used by both"],
        ["Invalid dot-com split", "The adaptive policy held 100 percent gold in every period", "GC=F history starts in August 2000, leaving 30 training days", "Documented as a failed setting; corrected setting uses VUSTX with 196 training days"],
        ["Run-to-run drift", "COVID-19 adaptive Sharpe moved from 0.85 to 1.05 with the same code and seed", "Small revisions in refreshed Yahoo data change the HMM fit", "Not removable; reported with seed ranges (Section 7.5)"],
    ], "Defects found during development and testing.", widths=[1.05, 1.7, 1.75, 1.9], font=8.5, align=["l"] * 4)

    # ======================================================================== CHAPTER 6
    R.chapter(6, "Project Demonstration")
    R.h2("6.1 Running the Application")
    R.p("The project needs Python 3.12 and Node.js, and it needs no API key. From the project folder one script installs "
        "the dependencies on first use and starts both servers.")
    R.code(["./run.sh", "# back end:  http://127.0.0.1:8000   (interactive API docs at /docs)",
            "# dashboard: http://127.0.0.1:5173"])
    R.p("The first comparison after the server starts takes about 25 seconds, because the matcher trains the encoder and "
        "builds the corpus. The page says so while it waits. Later comparisons take about a tenth of a second.")
    R.h2("6.2 Portfolio Backtest Tab")
    R.p("The user picks one of the three case studies and presses Run Backtest. {fig:ss_bt} shows the 2008 crisis. The upper "
        "chart gives the value of one unit invested in each strategy. The dashed line marks the end of the adaptive "
        "policy's training data and the red band is the crisis window. Below the chart, the metrics table can switch "
        "between the out-of-sample segment and the full period, and the last panel stacks the weights the adaptive policy "
        "chose at each rebalance. In this run the policy moved almost entirely into long Treasuries (TLT) in late 2008.")
    R.figure("ss_bt", D["small_img"](FIGD / "ss_backtest.png"), "Portfolio Backtest tab for the 2008 crisis.", width=5.4)
    R.h2("6.3 Regime Detection Tab")
    R.p("The upper panel colors each trading day of the chosen symbol and range by regime. {fig:ss_auto} shows SPY from 2019 "
        "to 2021. The red band is the COVID-19 crash and amber and green bands alternate before and after it. The "
        "panel beneath is the Historical Pattern Comparison. In this capture it is still computing, which is what the "
        "user sees on the first comparison after startup.")
    R.figure("ss_auto", D["small_img"](FIGD / "ss_regime_auto.png"), "Regime Detection tab for SPY, 2019 to 2021, with the comparison loading.", width=5.4)
    R.p("When the user selects the dates 15 January to 15 May 2020 and clicks the COVID-19 button, the report in "
        "{fig:ss_covid} appears. The 85-trading-day window is judged to closely resemble the COVID-19 crash with 59 percent "
        "confidence. The overlay chart draws the selected period (blue) against a real COVID-19 reference window (purple, "
        "dashed). Four figures sit under it: annualized return of -32.1 percent, annualized volatility of 51.9 percent, a "
        "maximum drawdown of -33.7 percent and the match confidence. The written insights state that the HMM labelled 32 "
        "percent of the days Bull, 38 percent Bear and 29 percent Crisis, and that similar historical windows were followed "
        "by an average 21-day return of +5.2 percent (standard deviation 3.2 percent, 8 windows). The footer reminds the "
        "reader that the analysis is illustrative and not investment advice.")
    R.figure("ss_covid", D["small_img"](FIGD / "ss_regime_covid.png"), "Regime Detection tab with the comparison report for the COVID-19 window.", width=5.4)
    R.p("If a period matches no event, the same panel states that the period tracks none of the three crises, shows the "
        "confidence for the normal class and gives the normal-regime baseline in place of an event overlay. The 2013 to 2015 "
        "SPY window is an example (Section 7.2).")
    R.h2("6.4 About Tab")
    R.p("The About tab in {fig:ss_about} explains what the application does, lists the stack, walks through the pipeline "
        "and states the scope limits. One thing in it is out of date: its description of the match step still mentions "
        "nearest cluster centers, whereas the final scoring uses the nearest labelled windows (Section 5.1.5).")
    R.figure("ss_about", D["small_img"](FIGD / "ss_about.png"), "About and tech stack tab.", width=4.9)

    # ======================================================================== CHAPTER 7
    R.chapter(7, "Result and Discussion (Cost Analysis as Applicable)")
    R.h2("7.1 Regime Detection")
    R.p("The unsupervised HMM marks the long turbulent stretches of all three crises as Crisis ({fig:regimes}). The red "
        "band covers the 2008 and 2009 collapse, the low of October 2002 and the March 2020 crash, and the model was never "
        "told these dates. Crisis days are a minority of each window (19, 14 and 8 percent), and Bear states fill the slower "
        "phases before and after. In calm periods the labels flip between Bull and Bear more often, so we treat the HMM as "
        "a descriptive overlay and an input to the allocator, not as a trading signal.")
    R.figure("regimes", FIGD / "fig_regimes.png", "HMM regimes on SPY for (a) the dot-com bust, (b) the 2008 crisis and (c) the COVID-19 crash.", width=6.2)

    R.h2("7.2 Historical-Pattern Comparison")
    R.h3("7.2.1 Example Reports")
    R.p(f"{{tab:t_queries}} and {{fig:overlays}} show the report for five queries. The three crisis windows are matched to their "
        f"own events with confidences of {pct(q['COVID-19 crash window']['targeted']['confidence'], 0)} (COVID-19), "
        f"{pct(q['2008 GFC window']['targeted']['confidence'], 0)} (2008) and "
        f"{pct(q['Dot-com bust window']['targeted']['confidence'], 0)} (dot-com). The 2013 to 2015 period and the most "
        "recent months are labelled normal. These queries overlap the periods that were used to build the corpus, so the "
        "matches are partly circular. They show what the report looks like, and the cross-validation below measures how "
        "well the method generalizes.")
    rows = []
    for name, v in q.items():
        t = v["targeted"]
        res_txt = (f"{t['event']} ({pct(t['confidence'], 0)})" if v["target"]
                   else f"{'normal' if not v['auto']['event'] else v['auto']['event']} ({pct(v['auto']['confidence'], 0)})")
        rows.append([name, f"{v['symbol']} {v['start']} to {v['end']}", v["days"], res_txt, pct(v["stats"]["ann_return"]),
                     pct(v["stats"]["max_dd"]),
                     f"{pct(v['stats']['regime_mix']['bull'], 0)} / {pct(v['stats']['regime_mix']['bear'], 0)} / {pct(v['stats']['regime_mix']['crisis'], 0)}",
                     pct(v["fwd21"], 1)])
    R.table("t_queries", ["Query", "Symbol and range", "Days", "Result (confidence)", "Ann. return", "Max DD", "HMM bull / bear / crisis", "Next-21d baseline"], rows,
            "Comparison report output for five example queries.", widths=[0.95, 1.35, 0.4, 0.95, 0.6, 0.55, 0.9, 0.7], font=8,
            align=["l", "l", "c", "c", "c", "c", "c", "c"],
            note="Crisis rows are scored in targeted mode against the matching event, the other two in auto mode. The last column is the mean 21-day forward return of the nearest same-label reference windows.")
    R.figure("overlays", FIGD / "fig_overlays.png", "Query paths (solid) against the nearest reference windows (dashed).", width=6.2)
    R.h3("7.2.2 Embedding Space")
    R.p("{fig:embedding} projects the 16-dimensional embeddings to two dimensions. Normal windows form the dense bulk. "
        "Event windows occupy a separate but overlapping region in which dot-com, 2008 and COVID-19 windows are mixed. That "
        "mixing is what we would expect if a crash followed by a recovery has a similar shape in every crisis.")
    R.figure("embedding", FIGD / "fig_embedding.png", f"Corpus embeddings colored by label: (a) PCA and (b) t-SNE of the {ncorp} vectors.", width=6.0)
    R.h3("7.2.3 Leakage-Controlled Evaluation")
    R.p(f"{{tab:t_cv}} gives the blocked cross-validation. All variants reach about 75 to 76 percent accuracy, but almost all "
        f"of it comes from the normal class, which has 99 percent recall. The trained Transformer reaches a balanced "
        f"accuracy of {TR['balanced_acc']:.2f} against a chance level of 0.25. Dot-com windows are recognized in "
        f"{pct(TR['recall']['dotcom'], 0)} of cases, while 2008 and COVID-19 windows almost never are (recall "
        f"{pct(TR['recall']['gfc'], 0)} and {pct(TR['recall']['covid'], 0)}). The confusion matrix in {{fig:confusion}} shows "
        f"that they are mostly assigned to dot-com or to normal. Telling crisis from normal works better: "
        f"{pct(TR['crisis_recall'], 0)} of event windows are predicted as some crisis, with a false-alarm rate of "
        f"{pct(TR['crisis_fpr'])}. So on unseen time blocks the representation captures \"crisis-like\" but not \"which crisis\".")
    rows = []
    for n_, v in cv.items():
        rows.append([n_, f"{v['balanced_acc']:.3f} ± {v['balanced_acc_sd']:.3f}", f"{v['macro_f1']:.3f} ± {v['macro_f1_sd']:.3f}",
                     pct(v["acc"], 1), pct(v["crisis_recall"], 1), pct(v["crisis_fpr"], 1),
                     f"{pct(v['recall']['dotcom'], 0)} / {pct(v['recall']['gfc'], 0)} / {pct(v['recall']['covid'], 0)} / {pct(v['recall']['normal'], 0)}"])
    R.table("t_cv", ["Features (8-NN)", "Balanced acc.", "Macro-F1", "Accuracy", "Crisis recall", "False alarm", "Recall dot-com / 2008 / COVID / normal"], rows,
            "Blocked, purged 5-fold cross-validation of event identification.", widths=[1.45, 0.85, 0.85, 0.6, 0.6, 0.6, 1.45], font=8,
            align=["l", "c", "c", "c", "c", "c", "c"], bold_rows=(0,),
            note="Mean ± standard deviation over three encoder seeds for the two Transformer variants; the other variants are deterministic. Chance balanced accuracy is 0.25.")
    R.figure("cv", FIGD / "fig_cv_ablation.png", "Embedding ablation under blocked cross-validation.", width=6.0)
    R.figure("confusion", FIGD / "fig_confusion.png", "Confusion matrix of the trained-Transformer classifier under blocked cross-validation.", width=3.4)
    R.h3("7.2.4 Unseen Crises")
    R.p(f"In leave-one-event-out tests ({{tab:t_loeo}}) the trained encoder flagged "
        f"{pct(lo['dotcom']['Transformer (trained)']['crisis_like_rate'], 0)} of dot-com windows, "
        f"{pct(lo['gfc']['Transformer (trained)']['crisis_like_rate'], 0)} of 2008 windows and "
        f"{pct(lo['covid']['Transformer (trained)']['crisis_like_rate'], 0)} of COVID-19 windows as crisis-like, although no "
        "window of that crisis was available for training. The 2008 crisis was a slow decline whose short sub-windows look "
        "like ordinary periods, which explains why it is the hardest. The result supports using the tool to ask whether a "
        "period looks like a crisis in general.")
    rows = []
    for ev, nm in (("dotcom", "Dot-com"), ("gfc", "2008 crisis"), ("covid", "COVID-19")):
        r = [nm, lo[ev]["n_test"], lo[ev]["n_train"]]
        for v in cv:
            r.append(pct(lo[ev][v]["crisis_like_rate"], 0))
        rows.append(r)
    R.table("t_loeo", ["Held-out event", "Test", "Train", "Transformer (trained)", "Transformer (untrained)", "Raw shape", "PCA-16", "Summary stats"], rows,
            "Leave-one-event-out: share of the unseen crisis' windows predicted as crisis-like.",
            widths=[0.95, 0.45, 0.5, 0.95, 0.95, 0.7, 0.65, 0.8], font=8.5, align=["l"] + ["c"] * 7)

    R.h2("7.3 Walk-Forward Allocation Backtests")
    bd, bc, bg, bv = (BT["cases"][k]["metrics_test"] for k in ("dotcom_deployed", "dotcom_corrected", "gfc", "covid"))
    R.p(f"{{tab:t_bt}} and {{fig:curves}} to {{fig:bars}} give the out-of-sample results. The picture is mixed and depends "
        f"heavily on the experimental design.")
    R.p(f"**Dot-com as deployed.** The adaptive portfolio reaches a Sharpe ratio of {bd['adaptive']['sharpe_ratio']:.2f} and a "
        f"maximum drawdown of {pct(bd['adaptive']['max_drawdown'])}, ahead of both baselines. That result is not real. Its "
        "weights are 100 percent gold at all 40 rebalances and its turnover is zero. The gold futures series starts in August "
        "2000, so the training segment is only 30 trading days and the policy collapses to a buy-and-hold gold position, "
        "which happened to rise.")
    R.p(f"**Dot-com corrected** (SPY, QQQ and VUSTX, 196 genuine pre-crisis training days). Now the adaptive policy loses "
        f"badly, with a Sharpe ratio of {bc['adaptive']['sharpe_ratio']:.2f} and a maximum drawdown of "
        f"{pct(bc['adaptive']['max_drawdown'])}. A policy trained in the bubble years concentrates in QQQ and then flips "
        "between QQQ and bonds ({fig:alloc}).")
    R.p(f"**2008 crisis.** Mean-variance has the best Sharpe ratio ({bg['mean_variance']['sharpe_ratio']:.2f}). The adaptive "
        f"policy has the lowest volatility ({pct(bg['adaptive']['annualized_volatility'])}) but not the shallowest drawdown. "
        f"**COVID-19.** Mean-variance leads again ({bv['mean_variance']['sharpe_ratio']:.2f}), and the adaptive policy "
        f"({bv['adaptive']['sharpe_ratio']:.2f}) is close to equal-weight ({bv['equal_weight']['sharpe_ratio']:.2f}).")
    rows, bold = [], []
    for k in CASES:
        c = BT["cases"][k]
        best = max(c["metrics_test"], key=lambda s: c["metrics_test"][s]["sharpe_ratio"])
        for s in ("equal_weight", "mean_variance", "adaptive"):
            m = c["metrics_test"][s]
            rows.append([CNAME[k] if s == "equal_weight" else "", SNAME[s], pct(m["total_return"]), pct(m["annualized_return"]),
                         pct(m["annualized_volatility"]), f"{m['sharpe_ratio']:.2f}", pct(m["max_drawdown"]), f"{m['calmar_ratio']:.2f}",
                         f"{c['turnover'][s]:.2f}"])
            if s == best:
                bold.append(len(rows) - 1)
    R.table("t_bt", ["Setting", "Strategy", "Total return", "Ann. return", "Ann. vol.", "Sharpe", "Max DD", "Calmar", "Turnover"], rows,
            "Out-of-sample performance of the three strategies (policy seed 42).", widths=[1.15, 0.95, 0.65, 0.65, 0.6, 0.5, 0.6, 0.5, 0.6],
            font=8.5, align=["l", "l"] + ["c"] * 7, bold_rows=tuple(bold),
            note="The row with the highest Sharpe ratio in each setting is in bold. Turnover is the mean weight change per monthly rebalance; equal-weight targets never change, so it reports 0.")
    R.figure("curves", FIGD / "fig_backtest_curves.png", "Portfolio value of the three strategies in the four settings.", width=6.2)
    R.figure("bars", FIGD / "fig_metrics_bars.png", "Out-of-sample Sharpe ratio, maximum drawdown and volatility.", width=6.2)
    R.figure("alloc", FIGD / "fig_allocations.png", "Adaptive-policy weights over time (seed 42).", width=6.2)

    R.h3("7.3.1 Criterion-Wise Analysis")
    crit = [("Total return", "total_return", max), ("Annualized volatility (lowest)", "annualized_volatility", min),
            ("Sharpe ratio", "sharpe_ratio", max), ("Max drawdown (shallowest)", "max_drawdown", max), ("Calmar ratio", "calmar_ratio", max)]
    ab = {"equal_weight": "EW", "mean_variance": "MV", "adaptive": "AD"}
    rows = []
    for nm, key, fn in crit:
        r = [nm]
        for k in CASES:
            ms = BT["cases"][k]["metrics_test"]
            r.append(ab[fn(ms, key=lambda s: ms[s][key])])
        rows.append(r)
    R.table("t_crit", ["Criterion"] + [CNAME[k] for k in CASES], rows, "Best strategy per criterion and setting on the test segment.",
            widths=[2.0, 1.1, 1.1, 1.1, 1.1], font=9.5, align=["l", "c", "c", "c", "c"],
            note="EW is Equal-Weight, MV is Mean-Variance, AD is Adaptive.")
    R.p("Mean-variance is best on return, Sharpe ratio, Calmar ratio and drawdown in three of the four settings. The adaptive "
        "policy has the lowest volatility in three settings, which is the quantity its reward penalizes through the drawdown and "
        "turnover terms, but Section 7.5 shows that this does not hold across seeds. Equal-weight is never best and has the "
        "deepest drawdown in three of four settings, yet it is the cheapest to run, with no optimization and no change in "
        "target weights.")

    R.h2("7.4 Statistical Analysis")
    rows = []
    for k in CASES:
        b = BT["cases"][k]["bootstrap"]
        for base, lab in (("mean_variance", "vs Mean-Variance"), ("equal_weight", "vs Equal-Weight")):
            d = b[f"adaptive_minus_{base}"]
            rows.append([CNAME[k] if base == "mean_variance" else "", lab, f"{d['sharpe_diff']:+.2f}", f"[{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]", f"{d['p_diff_le_0']:.2f}"])
    R.table("t_boot", ["Setting", "Adaptive minus", "Sharpe difference", "95% bootstrap CI", "P(difference ≤ 0)"], rows,
            "Paired stationary-bootstrap intervals for the test-segment Sharpe ratio difference.", widths=[1.5, 1.4, 1.0, 1.5, 1.0], font=9.5,
            align=["l", "l", "c", "c", "c"])
    sd_max = max(v["balanced_acc_sd"] for v in cv.values())
    R.p("Every interval contains zero. With test segments of 473 to 957 daily returns, none of the observed Sharpe ratio "
        "differences can be told apart from zero in either direction, so neither the dot-com \"win\" nor the other \"losses\" "
        "should be read as evidence about the method. Differences between seeds are a larger effect than most of the "
        "differences between strategies (Section 7.5). The standard deviation of balanced accuracy across encoder seeds is "
        f"at most {sd_max:.3f}, so the matcher results are stable even though the allocator results are not. We report eight "
        "comparisons without a multiplicity correction. Since no interval excludes zero, a correction would not change any conclusion.")

    R.h2("7.5 Ablation Study")
    R.p("{tab:t_abl} and {fig:seedabl} give the mean and standard deviation of the test-segment results over six seeds. "
        "Removing the regime features does not change the outcome in any consistent direction. The 2008 Sharpe ratio drops "
        "slightly and the COVID-19 one rises, both within one standard deviation. A more telling result is that the "
        "untrained random policy matches or beats the trained one in the 2008 and COVID-19 settings (Sharpe 0.59 against "
        "0.36 and 1.02 against 0.90), and has lower volatility in COVID-19 (14.1 against 19.2 percent). Training by "
        "evolution strategies on the short pre-crisis segment overfits and does not carry over to the crisis. In the "
        "as-deployed dot-com setting the trained variants give identical results for every seed because they all converge "
        "to the same constant gold portfolio.")
    rows = []
    for k in CASES:
        sub = abl[abl.case == k]
        s0 = sub[(sub.cond == "full") & (sub.seed == 0)].iloc[0]
        rows.append([CNAME[k], "Equal-Weight", f"{s0.equal_weight_sharpe:.2f}", pct(s0.equal_weight_vol), pct(s0.equal_weight_mdd), "n/a"])
        rows.append(["", "Mean-Variance", f"{s0.mean_variance_sharpe:.2f}", pct(s0.mean_variance_vol), pct(s0.mean_variance_mdd), "n/a"])
        floor = min(s0.equal_weight_vol, s0.mean_variance_vol)
        for c, lab in (("full", "Adaptive (full)"), ("no_regime", "Adaptive (no regime)"), ("untrained", "Adaptive (untrained)")):
            x = sub[sub.cond == c]
            rows.append(["", lab, f"{x.adaptive_sharpe.mean():.2f} ± {x.adaptive_sharpe.std():.2f}",
                         f"{pct(x.adaptive_vol.mean())} ± {pct(x.adaptive_vol.std())}",
                         f"{pct(x.adaptive_mdd.mean())} ± {pct(x.adaptive_mdd.std())}", f"{(x.adaptive_vol < floor).mean() * 100:.0f}%"])
    R.table("t_abl", ["Setting", "Strategy or variant", "Sharpe", "Ann. volatility", "Max drawdown", "Seeds with lowest volatility"], rows,
            "Allocator ablation: test-segment results, mean ± standard deviation over six seeds.",
            widths=[1.15, 1.35, 1.0, 1.15, 1.15, 0.95], font=8, align=["l", "l", "c", "c", "c", "c"],
            shade_rows=tuple(range(0, len(rows), 5)),
            note="Baselines are deterministic. The last column is the share of seeds in which the adaptive variant has lower volatility than both baselines.")
    R.figure("seedabl", FIGD / "fig_seed_ablation.png", "Out-of-sample Sharpe ratio of the adaptive policy across six seeds; dashed lines are the baselines.", width=6.2)
    R.p("For the matcher, replacing the trained Transformer with an untrained one, the raw shape, PCA or seven summary "
        "statistics changes balanced accuracy by at most 0.022 ({tab:t_cv}). The leave-one-event-out rates ({tab:t_loeo}) are "
        "similar: summary statistics are highest for COVID-19 (96 against 93 percent for the trained encoder), the untrained "
        "encoder is highest for dot-com (96 against 90 percent), and the trained encoder is above the raw features for "
        "dot-com and COVID-19. The useful signal sits in the resampled shape itself, and at this data scale the Transformer "
        "neither adds to it nor removes any of it.")

    R.h2("7.6 Comparison with Existing Methods")
    R.p("A number-for-number comparison with the four reviewed studies is not possible. They use different data, assets, "
        "periods and costs, and their code and data were not available to us. We therefore compare against equal-weight and "
        "mean-variance under one protocol, and compare capabilities with the reviewed approaches in {tab:t_compare}, using only "
        "the properties that those studies report.")
    R.table("t_compare", ["Capability", "Mean-Variance / 1/N", "Zhang (GAT)", "DRL-APO", "RL-APM", "Regime-aware RL", "This work"], [
        ["Multi-asset allocation", "Yes", "Yes", "Yes", "No (single stock)", "Yes", "Yes (3 to 4 assets)"],
        ["Explicit regime detection", "No", "Not reported", "No", "No", "Yes", "Yes (HMM)"],
        ["Transaction costs in objective", "No", "Not reported", "Not reported", "Ignored", "Not reported", "Yes (turnover and 5 bps)"],
        ["Historical-analogue comparison", "No", "No", "No", "No", "No", "Yes (Transformer and HDBSCAN)"],
        ["Interpretable output", "Yes", "Low", "Low", "Medium", "Medium (LSTM)", "Yes (report and overlay)"],
        ["Leakage-controlled, multi-seed evaluation", "n/a", "Not reported", "Not reported", "Not reported", "Not reported", "Yes"],
        ["Reported to beat static baselines", "n/a", "Yes (reported)", "Yes (reported)", "Yes (reported)", "Yes (reported)", "No (this report)"],
    ], "Capability comparison with the reviewed studies.", widths=[1.45, 0.85, 0.8, 0.7, 0.75, 0.85, 1.1], font=8, align=["l"] + ["c"] * 6,
        note="\"Not reported\" means the literature review did not record the property, not that the study lacks it.")
    R.p("The difference lies in scope and in how the results are reported. The application adds an explainable comparison and "
        "an evaluation with seeds, ablations and leakage controls. It does not show that its allocator beats the baselines.")

    R.h2("7.7 Cost Analysis")
    R.p("The monetary cost of the project was nil (Section 3.2.2). {tab:t_runtime} gives the computing cost on the development "
        "laptop. Training and scoring are cheap because the Transformer and the corpus are small.")
    R.table("t_runtime", ["Operation", "Time on Apple M3 (CPU)"], [
        ["Build the matcher (train encoder, embed 965 windows, HDBSCAN grid), once per server start", "about 25 s"],
        ["One comparison request after the matcher exists", "about 0.1 s"],
        ["One backtest including HMM fit and 40 ES generations", "8 to 17 s"],
    ], "Computing cost of the main operations.", widths=[4.7, 1.7], font=10, align=["l", "l"])
    R.p("The trading cost inside the backtests is modelled at 5 basis points of turnover. Mean-Variance trades heavily "
        f"({BT['cases']['gfc']['turnover']['mean_variance']:.2f} per rebalance in 2008, {BT['cases']['covid']['turnover']['mean_variance']:.2f} in COVID-19), "
        "and pays for it in return relative to a frictionless run. TC15 in {tab:t_unittests} confirms that a higher cost lowers net return.")

    R.h2("7.8 Discussion")
    R.bullets([
        "**Regimes can be found without labels, but the label is not a forecast.** The HMM recovers the turbulent phases of all "
        "three crises. Its day-level states are noisy, and removing them from the allocator's input made no consistent difference.",
        "**Shape similarity detects crises and cannot yet name them.** Once windows are reduced to normalized shapes, the three "
        "crises look alike, so telling them apart on held-out blocks fails while crisis against normal works with few false "
        "alarms. The confidence in the report is closeness to a family of crash shapes. It is not the probability that the "
        "period is that event.",
        "**The Transformer is not where the signal comes from.** An untrained encoder, the raw shape and seven summary "
        "statistics do as well. With under a thousand strongly overlapping windows this is expected, and the encoder acts "
        "close to a fixed random projection that preserves distances.",
        "**The adaptive allocator did not beat simple baselines.** Mean-variance is the strongest or near strongest in three of "
        "four settings, and apparent wins were fragile.",
        "**The experimental design can decide the conclusion.** Swapping gold futures for a bond fund in the dot-com setting "
        "turned the best strategy into the worst.",
    ])
    R.h2("7.9 Limitations")
    R.bullets([
        "Three crises and four calm periods give 965 overlapping windows with a large normal class. Blocked purging reduces "
        "the dependence between windows but cannot remove it, and other crises (1987, 1998, 2022) were not tested.",
        "The example queries in {tab:t_queries} overlap the corpus; only the blocked and leave-one-event-out results measure generalization.",
        "Evolution strategies replace the PPO and DDPG agents of the literature, with a small state, 40 generations and short "
        "training segments. The negative result applies to this implementation and says little about deep reinforcement "
        "learning in general. Only long-only portfolios of three or four assets were studied, with one cost assumption and "
        "no liquidity or market-impact model, which was one of the original objectives.",
        "The as-deployed dot-com setting is invalid and is kept only to document the diagnosis. The corrected setting uses a "
        "mutual fund as a proxy for bonds, and each test segment contains a single crisis.",
        "Results move slightly when Yahoo refreshes its data, as the COVID-19 case showed. The endpoint is unofficial and may "
        "change, and adjusted prices are revised retroactively.",
    ])

    # ======================================================================== CHAPTER 8
    R.chapter(8, "Conclusion")
    R.p("We built and evaluated a web application that compares any market period with the dot-com bust, the 2008 crisis and "
        "the COVID-19 crash. It combines a length-invariant shape representation, a Transformer encoder and nearest-neighbor "
        "scoring over an HDBSCAN-structured corpus with an HMM regime view and a walk-forward allocation workbench. The "
        "evaluation gives a clear but modest result. The comparison separates crisis-like from normal periods "
        f"(crisis recall {pct(TR['crisis_recall'], 0)} at a {pct(TR['crisis_fpr'])} false-alarm rate) and recognizes unseen "
        "crises as crisis-like. It cannot say which crisis a period resembles, and neither the trained Transformer nor the "
        "regime-aware allocator beat simpler alternatives.")
    R.p("{tab:t_status} compares the outcome with the objectives from Section 2.3.")
    R.table("t_status", ["Objective", "Outcome", "Status"], [
        ["1. Adaptive allocation model for multi-asset portfolios", "Implemented as an evolution-strategies policy rather than PPO or DDPG; works on 3 to 4 assets", "Partly met"],
        ["2. Real-time regime detection to trigger reallocation", "HMM regimes feed the policy state; ablation shows no consistent benefit", "Met, benefit not shown"],
        ["3. Transaction costs and liquidity in the reward", "Turnover cost and drawdown in the reward; no liquidity model", "Partly met"],
        ["4. Benchmark against mean-variance and equal-weight", "Sharpe, drawdown, return, volatility and Calmar reported, with bootstrap intervals", "Met"],
        ["5. Robustness across bull, bear and volatile regimes", "Four settings, six seeds and ablations; no consistent advantage found", "Met, result negative"],
        ["6. Historical comparison with Transformer and HDBSCAN", "Working report with overlay, statistics and baseline; evaluated with leakage controls", "Met"],
        ["7. Working web application on live data", "Eight endpoints, three-tab dashboard, 21 of 21 tests passed", "Met"],
    ], "Objectives and outcomes.", widths=[2.4, 3.1, 0.9], font=9.5, align=["l", "l", "c"])
    R.p("We regard the diagnostics as useful results in their own right: the invalid split that produced a constant gold "
        "portfolio, the untrained-encoder and untrained-policy controls, and the seed and bootstrap analysis.")
    R.h2("8.1 Future Work")
    R.bullets([
        "Extend the corpus to more crises and markets, and cross-validate by whole event instead of by window.",
        "Replace the single-crisis labels by a continuous severity score, and add shape descriptors that keep the original "
        "duration so that fast and slow crashes can be told apart.",
        "Pre-train the encoder on a much larger set of unlabelled windows, or try contrastive objectives, before claiming any "
        "benefit from attention.",
        "Implement PPO and DDPG agents under the same protocol with longer training windows, rolling retraining, policy "
        "regularization and a cost and liquidity model.",
        "Add automated tests to the repository and make the dot-com configuration valid by default.",
        "Move toward TRL 4 and 5 with paper trading on live data, and evaluate the dashboard with real users.",
    ])

    # ======================================================================== CHAPTER 9
    R.chapter(9, "References")
    R.reference_list()

    # ======================================================================== APPENDIX A
    from report_lib import set_run_font
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    p = R._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=10, keep_next=True)
    p.paragraph_format.page_break_before = True
    from report_tx import tx_text
    r = p.add_run(tx_text("APPENDIX A: SAMPLE CODE", "h")); set_run_font(r, 14, bold=True)
    R._outline(p, 0)
    R.headings.append((0, "A|APPENDIX A: SAMPLE CODE", R._bookmark(p)))
    app = ROOT / "backend" / "app"
    parts = [
        ("A.1 Transformer shape encoder (ml/embedding.py)", app / "ml" / "embedding.py", {"ShapeEncoder"}),
        ("A.2 Length-invariant shape (ml/embedding.py)", app / "ml" / "embedding.py", {"build_channels", "resample_to_fixed"}),
        ("A.3 Label scoring (ml/event_matcher.py)", app / "ml" / "event_matcher.py", {"_score_label"}),
        ("A.4 Mean-variance weights and adaptive policy (ml/allocator.py)", app / "ml" / "allocator.py", {"mean_variance_weights", "PolicyNet"}),
        ("A.5 Evolution strategies training loop (ml/allocator.py)", app / "ml" / "allocator.py", {"train_adaptive_policy"}),
        ("A.6 Comparison route (main.py)", app / "main.py", {"regimes_compare"}),
    ]
    for title, path, names in parts:
        R.h3(title)
        segs = extract_functions(path, names)
        for seg in segs:
            R.code(wrap_code(seg))
    R.h3("A.7 Commands to reproduce the experiments")
    R.code(["./run.sh                                             # API and dashboard",
            "backend/.venv/bin/python paper/scripts/exp_backtest.py   # backtests, bootstrap",
            "backend/.venv/bin/python paper/scripts/exp_ablation.py   # allocator ablation",
            "backend/.venv/bin/python paper/scripts/exp_matcher.py    # matcher CV and LOEO",
            "backend/.venv/bin/python paper/scripts/run_tests.py      # 21 test cases"])
