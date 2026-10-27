"""Chapters 1 to 4 of the project report."""
FIG = None  # set by build_report


def write(R, D):
    FIGD = D["FIG"]
    # ======================================================================== CHAPTER 1
    R.chapter(1, "Introduction")
    R.h2("1.1 Background")
    R.p("Prices in a financial market do not follow one fixed pattern. A calm year, a slow bear market and a sudden crash "
        "each have their own level of volatility, and assets that usually move independently tend to fall together when "
        "a crisis arrives {cite:cont01}. Three episodes from the last twenty-five years show how different these events can "
        "be. In the dot-com bust (March 2000 to October 2002) the Nasdaq-100, which we follow through the QQQ fund, lost "
        "about 78 percent of its value over two and a half years. In the 2008 global financial crisis (October 2007 to "
        "March 2009) the S&P 500 fell by about 57 percent from peak to trough. The COVID-19 crash was far shorter: the "
        "S&P 500 dropped roughly 34 percent between 19 February and 23 March 2020 and then recovered faster than in any "
        "earlier bear market.")
    R.p("Most portfolio construction still starts from the mean-variance framework of Markowitz {cite:markowitz52}, which "
        "picks asset weights from the average returns and covariances observed in a past window. Those estimates are noisy, "
        "and when the market changes character the window is already out of date. DeMiguel and co-authors "
        "{cite:demiguel09} found that splitting money equally across assets, the 1/N rule, is hard to beat out of sample, "
        "which tells us how little the estimated inputs can be trusted. Two lines of research try to fix this. Regime-switching "
        "models {cite:hamilton89,angbekaert02} let the data choose between a few market states, and reinforcement learning "
        "{cite:moody01,jiang17} lets an agent learn an allocation rule directly from prices.")
    R.p("This project sits where those ideas meet. We built a web application that takes live price data, labels the "
        "current market regime, compares any period the user selects with the three crises above, and tests whether an "
        "adaptive allocation rule beats simple benchmarks. During development we made the comparison with past crises the "
        "central feature, because it answers a question people ask in practice (\"does this look like 2008?\") and can be "
        "checked against evidence.")

    R.h2("1.2 Motivations")
    R.p("Several things pushed us toward this topic.")
    R.bullets([
        "People reason about the present by analogy with past crises, but the analogy is usually informal. A tool can say "
        "how close a period is to each known crisis, on what evidence, and can say plainly when it resembles none of them.",
        "Static allocation rules assume stable market conditions. The reviewed papers (Table 2.1) report that adaptive "
        "methods do better, but few of them include transaction costs, repeat the training with several random seeds or "
        "guard against information leaking from the test period. We wanted to see how much of the claimed gain survives a "
        "stricter test.",
        "Free daily price data now goes back decades, which makes it possible to study all three crises with the same "
        "code and to run the application on live data.",
        "The project let us combine time series modelling, deep learning, clustering, numerical optimization and web "
        "development in one system, which is how such tools are built outside the classroom.",
    ])

    R.h2("1.3 Scope of the Project")
    R.p("The project covers daily data for liquid exchange-traded funds and a few other instruments. It delivers a back end, "
        "a dashboard and an evaluation of the models behind them. Table 1.1 lists what the work includes and what it leaves "
        "out.")
    R.table("t_scope", ["Within the scope", "Outside the scope"], [
        ["Daily adjusted close prices from Yahoo Finance, fetched live and cached", "Intraday, tick-level or order-book data"],
        ["Three historical case studies: dot-com bust, 2008 financial crisis, COVID-19 crash", "Other crises such as 1987, 1998 or 2022, and non-US markets"],
        ["Unsupervised regime labelling with a three-state hidden Markov model", "Using the regime label as a trading signal or forecast"],
        ["Comparison of any window of at least 20 trading days with the case studies, using a Transformer encoder and HDBSCAN", "Predicting future prices or returns"],
        ["Long-only portfolios of three or four assets, rebalanced every 21 trading days with a 5 basis point cost", "Short selling, leverage, derivatives, taxes and a liquidity or market-impact model"],
        ["Walk-forward backtests, bootstrap intervals, seed and feature ablations", "Live or paper trading with real money"],
        ["A browser dashboard that runs on a single machine", "Multi-user deployment, authentication and cloud hosting"],
    ], "Scope of the project.", widths=[3.2, 3.2], font=10, align=["l", "l"])
    R.p("The work is a proof of concept at technology readiness level 3, as set out in Section 2.3. It is an educational and "
        "research tool. Nothing in the application or in this report is investment advice.")

    # ======================================================================== CHAPTER 2
    R.chapter(2, "Project Description and Goals")
    R.h2("2.1 Literature Review")
    R.p("We began with four recent studies on adaptive portfolio optimization, listed in Table 2.1. All four report that "
        "an adaptive reinforcement learning agent does better than static allocation, but they differ in architecture, "
        "assets and how much they say about costs and interpretability.")
    R.table("t_lit", ["Paper", "Objective", "Methodology", "Pros and cons", "Findings"], [
        ["Zhang (2025), GAT multi-agent DRL {cite:zhang25}", "Adapt to time-varying asset correlations", "Graph attention network with three specialized RL agents", "High returns; high complexity", "Outperforms mean-variance and equal-weight"],
        ["DRL-APO framework (2025) {cite:drlapo25}", "Adaptive optimization for dynamic markets", "CNN-LSTM encoder with PPO policy gradient", "Captures temporal patterns; price-only inputs", "More adaptive than static models"],
        ["RL-APM / DDPG (2025) {cite:rlapm25}", "Real-time single-stock allocation", "DDPG agent with technical indicators (SMA, RSI, MACD)", "Continuous control; ignores costs and liquidity", "Improves on static baselines"],
        ["Regime-aware RL (2025) {cite:regimerl25}", "Regime-adaptive, explainable allocation", "Transformer-PPO and LSTM with regime detection", "Best returns; LSTM more interpretable", "Regime context improves stability"],
    ], "Literature review of recent adaptive portfolio optimization studies.", widths=[1.25, 1.2, 1.4, 1.3, 1.2], font=9, align=["l"] * 5)
    R.p("Beyond these four, we read the work our methods rest on. Markowitz {cite:markowitz52} gave the mean-variance "
        "framework, and Sharpe {cite:sharpe94} defined the return-to-risk ratio we use to score portfolios. DeMiguel et al. "
        "{cite:demiguel09} compared fourteen optimized strategies with equal weighting and found that none consistently "
        "beat it, so we use both as baselines.")
    R.p("For regimes, Hamilton {cite:hamilton89} introduced Markov-switching models for series whose behavior changes "
        "between states, and Ang and Bekaert {cite:angbekaert02} showed that such regimes matter for asset allocation. "
        "Hidden Markov models {cite:rabiner89} are the standard way to fit these states without labels, and we use the "
        "hmmlearn implementation {cite:hmmlearn}.")
    R.p("Reinforcement learning for trading goes back to Moody and Saffell {cite:moody01}. Jiang et al. {cite:jiang17} "
        "applied deep networks to portfolio management, and DDPG {cite:lillicrap16} and PPO {cite:schulman17} are the usual "
        "training algorithms in the papers above. Salimans et al. {cite:salimans17} showed that evolution strategies, which "
        "estimate a gradient from the rewards of randomly perturbed parameters, can train policies without back-propagation. "
        "We chose evolution strategies for the allocator because they need no automatic differentiation and finish "
        "training in seconds on a laptop.")
    R.p("The Transformer {cite:vaswani17} relies on self-attention instead of recurrence. Zerveas et al. {cite:zerveas21} "
        "showed that Transformer encoders can learn representations of multivariate time series without labels, and we "
        "follow that idea at a small scale. HDBSCAN {cite:campello13,mcinnes17} clusters points of varying density and marks "
        "sparse points as noise, which suits the question of whether a window belongs to any known group at all. We use "
        "the scikit-learn implementation {cite:pedregosa11}.")

    R.h2("2.2 Research Gap")
    R.p("Comparing the four studies shows four gaps, summarized in Table 2.2 together with the focus we gave each one and how "
        "far the finished project went.")
    R.table("t_gap", ["Gap area", "Limitation in existing work", "Proposed focus of this work", "Status in this project"], [
        ["Model adaptivity", "Most RL models assume a single asset or a simplified setting and ignore transaction costs and liquidity constraints", "Multi-asset framework with transaction costs and liquidity in the reward function", "Multi-asset and transaction costs done; liquidity not modelled"],
        ["Regime awareness", "Static mean-variance and early RL models do not detect or respond to shifting regimes in real time", "Explicit regime-detection module feeding the allocation decision", "Done with an HMM; ablation shows no consistent gain from it"],
        ["Risk-adjusted evaluation", "Many studies report returns without risk-adjusted benchmarking across varied conditions", "Backtests with Sharpe ratio, maximum drawdown and volatility-based metrics", "Done, plus seeds, bootstrap intervals and ablations"],
        ["Interpretability versus performance", "High-performing deep RL models such as Transformer-PPO give up interpretability, which limits adoption", "Balance predictive performance with explainable decision outputs", "Regime view and comparison report are explainable; the allocation policy is not"],
    ], "Research gaps, proposed focus and status.", widths=[1.1, 2.0, 1.8, 1.6], font=9, align=["l"] * 4)
    R.p("A fifth gap emerged while reviewing the material. None of the studies offers a way to compare an arbitrary "
        "market period with named past crises and report how strong the resemblance is. We added that as a feature, and it "
        "became the part of the project we evaluated most carefully.")

    R.h2("2.3 Objectives")
    R.p("The aim set out at the first review was to design and evaluate an adaptive portfolio optimization framework that "
        "reallocates assets in response to market regime shifts and improves risk-adjusted returns over traditional static "
        "strategies. The framework should reduce reliance on the fixed distributional assumptions of mean-variance models, "
        "support real-time data-driven rebalancing validated by backtesting on historical multi-asset data, and be benchmarked "
        "and interpretable enough for portfolio managers and researchers to use. The objectives were:")
    R.bullets([
        "Develop an adaptive allocation model for multi-asset portfolios (proposed as a deep reinforcement learning model).",
        "Integrate real-time market regime detection to trigger dynamic portfolio reallocation.",
        "Incorporate transaction costs and liquidity constraints into the reward function for realistic decisions.",
        "Benchmark against static strategies (mean-variance and equal-weight) using Sharpe ratio, drawdown and returns.",
        "Validate robustness across bull, bear and volatile market regimes using historical financial data.",
    ], numbered=True)
    R.p("Two further objectives were added during development:")
    R.bullets([
        "Build a historical-pattern comparison that shows whether a chosen period follows the same trend as the dot-com "
        "bust, the 2008 crisis or the COVID-19 crash, using a Transformer encoder and HDBSCAN, and gives a normal-regime "
        "baseline when no crisis matches.",
        "Package everything as a working web application that uses live market data and explains its output.",
    ])
    R.p("**Sustainable Development Goals.** The project was mapped to SDG 8 and SDG 9 {cite:un2015}. Table 2.3 states "
        "how, and the link is indirect: the work is a research prototype and does not itself change any economic outcome.")
    R.table("t_sdg", ["Goal", "Connection to the project"], [
        ["SDG 8: Decent work and economic growth", "Clear, evidence-based risk tools help investors and fund managers avoid large losses in crises, which supports stable economic activity. The tool is free and runs on a laptop."],
        ["SDG 9: Industry, innovation and infrastructure", "The project applies machine learning to financial infrastructure and releases the code and evaluation scripts as open source for other researchers."],
    ], "Sustainable Development Goals addressed.", widths=[2.2, 4.2], font=10, align=["l", "l"])
    R.p("**Expected outcomes.** A conference paper or a journal article indexed in Scopus. A full manuscript with the "
        "complete evaluation has been prepared for that purpose.")
    R.p("**Technology readiness level.** The project is assigned TRL 3, analytical and experimental proof of concept "
        "{cite:mankins95}. The regime detection and adaptive rebalancing functions were validated through simulation on "
        "historical multi-asset data. They have not been tested in a live trading or production environment. The target for "
        "future work is TRL 4 to 5, validation in a relevant environment with live or paper-traded data.")

    R.h2("2.4 Problem Statement")
    R.p("Traditional portfolio models such as mean-variance optimization and static allocation assume fixed market "
        "conditions. Real markets are non-stationary and move between regimes, so these models adapt slowly when volatility "
        "and correlations change. We therefore study two connected problems on daily multi-asset price data.")
    R.bullets([
        "**Historical-analogue comparison.** Given any symbol and date range of at least 20 trading days, decide whether "
        "the price path resembles the dot-com bust, the 2008 crisis, the COVID-19 crash or an ordinary period. Report the "
        "evidence: a similarity confidence, a chart overlay with a real historical window, the return, volatility and "
        "drawdown of the period, its regime composition, and a normal-regime baseline when nothing matches.",
        "**Regime-aware allocation.** Decide whether a policy that observes regime probabilities and asset momentum and "
        "volatility improves risk-adjusted performance over equal-weight and rolling mean-variance portfolios when it is "
        "trained only on data before a crisis and evaluated through the crisis and recovery, with transaction costs included.",
    ])

    R.h2("2.5 Project Plan")
    R.p("The four-month plan presented at the first review (17 August 2026) has six tasks that overlap in time. "
        "{fig:gantt} shows the schedule and {tab:t_plan} gives the week of each task.")
    R.figure("gantt", FIGD / "fig_gantt.png", "Four-month project work plan.", width=6.2)
    R.table("t_plan", ["No.", "Task", "Start (week)", "Duration (weeks)", "End (week)"], [
        ["1", "Literature survey and problem formulation", "0", "2", "2"],
        ["2", "Data collection and preprocessing", "1", "3", "4"],
        ["3", "Model design and algorithm development", "3", "5", "8"],
        ["4", "Implementation and backtesting", "7", "5", "12"],
        ["5", "Performance evaluation and tuning", "11", "3", "14"],
        ["6", "Documentation and report writing", "13", "3", "16"],
    ], "Project plan with task durations.", widths=[0.5, 3.3, 0.9, 1.0, 0.8], font=10, align=["c", "l", "c", "c", "c"])

    # ======================================================================== CHAPTER 3
    R.chapter(3, "Technical Specification")
    R.h2("3.1 Requirements")
    R.h3("3.1.1 Functional")
    R.p("{tab:t_fr} lists what the system must do and where each requirement is implemented.")
    R.table("t_fr", ["ID", "Requirement", "Implemented by"], [
        ["FR1", "Show live quotes for SPY, QQQ, GLD, TLT, BTC-USD and the VIX index in a ticker ribbon", "GET /api/market/live; TickerRibbon"],
        ["FR2", "Fetch daily history for any symbol and range, cache it, retry on failure, and fall back to labelled synthetic data for the three case studies if the feed is unreachable", "market_data.py; synthetic_fallback.py"],
        ["FR3", "Classify each trading day of a symbol and range as bull, bear or crisis and plot the result over the price", "GET /api/regimes/history; RegimeExplorer"],
        ["FR4", "Report the current regime of a symbol with its probability", "GET /api/regimes/current"],
        ["FR5", "Compare a window of at least 20 trading days with the dot-com, 2008 and COVID-19 events and a normal baseline, returning confidence, overlay path, statistics and written insights", "GET /api/regimes/compare; event_matcher.py"],
        ["FR6", "Offer an auto-detect mode and a mode that scores the window against one chosen event", "event_id parameter; EventComparisonReport"],
        ["FR7", "Run a backtest of Equal-Weight, Mean-Variance and Adaptive strategies for a case study or for custom assets and dates", "POST /api/backtest; engine.py"],
        ["FR8", "Let the user set rebalance frequency, cost in basis points, and the training generations and population of the adaptive policy", "BacktestRequest schema"],
        ["FR9", "Display value curves, a metrics table (total and annualized return, volatility, Sharpe, maximum drawdown, Calmar) and allocation weights over time", "BacktestPanel"],
        ["FR10", "Flag results that come from synthetic fallback data", "synthetic_data field in the response"],
        ["FR11", "Validate inputs and return clear error messages for missing dates, short windows, unknown case studies and unreachable data", "main.py exception handling"],
        ["FR12", "Describe the method and the technology stack inside the application", "AboutPanel"],
    ], "Functional requirements.", widths=[0.5, 4.0, 1.9], font=9.5, align=["c", "l", "l"])

    R.h3("3.1.2 Non-Functional")
    R.table("t_nfr", ["Attribute", "Requirement", "Measured or observed"], [
        ["Performance", "A comparison on a cached matcher should feel instant; a backtest should finish within a few seconds to tens of seconds", "Comparison about 0.1 s after the one-time matcher build of about 25 s; backtest 8 to 17 s"],
        ["Reliability", "A rate limit or outage at the data provider must not stop the demonstration", "Retry with back-off, 15-minute disk cache, labelled synthetic fallback"],
        ["Reproducibility", "Reported numbers must be repeatable", "Fixed seeds; experiment scripts and saved outputs in paper/"],
        ["Usability", "One page with three tabs, no sign-up and no API key", "Single-page React dashboard, dark theme"],
        ["Interpretability", "Output should show its evidence", "Overlay with a real historical window, written insights, confidence with stated basis"],
        ["Portability", "Run on a common developer machine", "macOS tested; Python 3.12 and Node 22; no GPU needed"],
        ["Maintainability", "Separate concerns so that a model can be replaced", "Packages: data, ml, backtest; about 3,200 lines in total"],
        ["Security and privacy", "No personal data and no secrets", "Only public market data is fetched; no accounts. Cross-origin access is open because the tool is meant for local use"],
        ["Scalability", "Single-user use", "In-memory caches in one process; not built for many concurrent users"],
    ], "Non-functional requirements.", widths=[1.1, 2.7, 2.6], font=9.5, align=["l", "l", "l"])

    R.h2("3.2 Feasibility Study")
    R.h3("3.2.1 Technical Feasibility")
    R.p("Every component we needed exists as mature open-source software: scikit-learn and hmmlearn for the hidden Markov "
        "model and HDBSCAN, PyTorch for the Transformer encoder, SciPy for the mean-variance optimizer, FastAPI for the "
        "service layer and React for the interface. The Transformer is small (32 hidden units, two layers) and the "
        "reference corpus has under a thousand windows, so training takes seconds on a CPU. The whole pipeline, including "
        "a full set of experiments, ran on a single Apple M3 laptop with 8 GB of memory. The part that was less certain "
        "was data access: Yahoo Finance rate-limited our early requests, and we solved it with a direct client, caching "
        "and a labelled fallback (Section 5.2.5).")
    R.h3("3.2.2 Economic Feasibility")
    R.p("The project needed no cash outlay. {tab:t_cost} lists the resources and what each cost.")
    R.table("t_cost", ["Resource", "Source", "Cost"], [
        ["Development machine (Apple M3, 8 GB)", "Already owned by a team member", "No additional cost"],
        ["Python 3.12, FastAPI, PyTorch, scikit-learn, hmmlearn, SciPy, pandas, NumPy", "Open-source licences", "Free"],
        ["Node.js, React, Vite, Recharts, axios", "Open-source licences", "Free"],
        ["Daily market data", "Yahoo Finance public chart endpoint, no API key", "Free"],
        ["Hosting", "Local machine at 127.0.0.1", "Free"],
        ["GPU or cloud compute", "Not used", "None"],
        ["Team effort", "Three students over 16 weeks (Section 2.5)", "Student time"],
    ], "Resources and costs.", widths=[2.7, 2.4, 1.3], font=10, align=["l", "l", "l"])
    R.p("A production service would add recurring costs for a licensed real-time data feed, hosting and monitoring. We "
        "did not estimate those because the project is not meant for production at this stage.")
    R.h3("3.2.3 Social Feasibility")
    R.p("The tool helps students and analysts see how today's market compares with past crises, and it is open about its "
        "limits. That openness matters socially: a confident-looking score can lead a user to risk money they cannot "
        "afford to lose. The dashboard therefore labels its output as illustrative and not predictive, shows when data is "
        "synthetic, and states the confidence value with the evidence behind it. The application collects no personal "
        "data. The main risk is misuse of the output as investment advice, which the interface and this report both discourage.")

    R.h2("3.3 System Specification")
    R.h3("3.3.1 Hardware Specification")
    R.table("t_hw", ["Component", "Specification"], [
        ["Processor", "Apple M3, 8 cores"],
        ["Memory", "8 GB unified memory"],
        ["Storage", "Under 1 GB for the code, dependencies and cached price files"],
        ["GPU", "Not required; the Transformer runs on the CPU"],
        ["Network", "Internet access for Yahoo Finance (the case studies also run offline on cached or synthetic data)"],
        ["Client", "Any current desktop browser"],
    ], "Hardware used for development and testing.", widths=[1.6, 4.8], font=10, align=["l", "l"])
    R.h3("3.3.2 Software Specification")
    R.table("t_sw", ["Layer", "Software", "Version"], [
        ["Operating system", "macOS", "26.6.2"],
        ["Language (back end)", "Python", "3.12.7"],
        ["Web framework", "FastAPI with Uvicorn; Pydantic", "0.115.6; 0.34.0; 2.10.4"],
        ["Numerical computing", "NumPy, pandas, SciPy", "1.26.4, 2.2.3, 1.13.1"],
        ["Machine learning", "scikit-learn (HDBSCAN, PCA, t-SNE, k-NN), hmmlearn, PyTorch", "1.5.2, 0.3.3, 2.5.1"],
        ["Front end", "React, Vite, Recharts, axios", "19.2, 8.2, 3.10, 1.20"],
        ["Runtime (front end)", "Node.js", "22"],
        ["Data source", "Yahoo Finance chart endpoint, called through curl", "n/a"],
    ], "Software stack.", widths=[1.6, 3.3, 1.5], font=10, align=["l", "l", "l"])
    R.p("The requirements file also lists the yfinance package, but the code does not use it. The data client talks to the "
        "Yahoo chart endpoint directly so that retries and caching are under our control.")

    # ======================================================================== CHAPTER 4
    R.chapter(4, "Design Approach and Details")
    R.h2("4.1 System Architecture")
    R.p("The application has four layers, shown in {fig:arch}. The React dashboard runs in the browser and talks to a "
        "FastAPI service over eight REST endpoints. The service calls the analytics layer, which holds the hidden Markov "
        "model, the pattern matcher, the adaptive allocator and the backtest engine. The data layer fetches prices from "
        "Yahoo Finance, stores them in a CSV cache and, for the three case studies only, can fall back to a synthetic "
        "series calibrated to each crisis. The matcher is built once per server process, which takes about 25 seconds, "
        "and is reused afterwards. Backtest results are cached by a hash of their settings.")
    R.figure("arch", FIGD / "fig_architecture.png", "Layered architecture of the Adaptive Portfolio Optimizer.", width=5.6)
    R.table("t_api", ["Method and path", "Purpose", "Main parameters"], [
        ["GET /api/health", "Liveness check", "none"],
        ["GET /api/case-studies", "Definitions of the three crises (dates, assets, description)", "none"],
        ["GET /api/assets/default", "Default four-asset universe (SPY, QQQ, GLD, TLT)", "none"],
        ["GET /api/market/live", "Latest quotes for the ticker ribbon", "none"],
        ["GET /api/regimes/current", "Current regime and probabilities of a symbol", "symbol, lookback_days"],
        ["GET /api/regimes/history", "Daily regime labels and probabilities", "symbol, start, end"],
        ["GET /api/regimes/compare", "Historical-pattern comparison report", "symbol, start, end, event_id (optional)"],
        ["POST /api/backtest", "Walk-forward backtest of three strategies", "case_study or assets, start, end, rebalance_freq, cost_bps, es_generations, es_population"],
    ], "REST endpoints of the back end.", widths=[2.0, 2.5, 1.9], font=9.5, align=["l", "l", "l"])

    R.h2("4.2 Design")
    R.h3("4.2.1 Data Flow Diagram")
    R.p("{fig:dfd0} is the context diagram. The user sends a symbol, a date range and optional settings, and receives "
        "charts, a comparison report and backtest results. Yahoo Finance is the only external data source.")
    R.figure("dfd0", FIGD / "fig_dfd0.png", "Data flow diagram, level 0 (context diagram).", width=5.8)
    R.p("{fig:dfd1} breaks the system into five processes. Process 1.0 acquires and cleans prices and writes them to the "
        "price cache (D1). Process 2.0 fits the hidden Markov model and returns regime labels. Process 3.0 compares a "
        "window with the reference corpus and the fitted matcher stored in D2. Process 4.0 runs the backtest and keeps "
        "finished results in D3. Process 5.0 is the dashboard, which collects the user's input and presents all outputs.")
    R.figure("dfd1", FIGD / "fig_dfd1.png", "Data flow diagram, level 1.", width=5.8)
    R.h3("4.2.2 Use Case Diagram")
    R.p("{fig:usecase} shows the two actors. The user can view the market ribbon and regime charts, compare a date range "
        "with historical crises (optionally targeting one event), read the report, run backtests and inspect their "
        "allocations. Yahoo Finance is a secondary actor that supplies prices to the use cases that need them. The most "
        "important use case is the comparison: the user picks a symbol and dates, optionally clicks one of the three event "
        "buttons, and receives a verdict with its evidence.")
    R.figure("usecase", FIGD / "fig_usecase.png", "Use case diagram.", width=5.6)
    R.h3("4.2.3 Class Diagram")
    R.p("{fig:class} shows the main data structures and modules of the back end. The API module receives a BacktestRequest "
        "and calls the market data, regime, matcher and backtest modules. A Matcher object bundles the encoder, the "
        "scaling constants, the list of CorpusWindow records, their embeddings and the cluster summaries. PolicyNet is the "
        "small neural policy used by the adaptive strategy, and ShapeAutoencoder wraps the ShapeEncoder with a decoder that "
        "is used only for training.")
    R.figure("class", FIGD / "fig_class.png", "Class diagram of the back end.", width=5.9)
    R.h3("4.2.4 Sequence Diagram")
    R.p("{fig:seq1} follows one comparison request. The dashboard calls the compare route, which hands the work to the "
        "matcher. The matcher gets prices through the data layer, converts the window into a 40 by 3 shape, asks the "
        "Transformer for an embedding, scores the labels and returns a report that the dashboard draws.")
    R.figure("seq1", FIGD / "fig_seq_compare.png", "Sequence diagram of a historical-pattern comparison.", width=5.9)
    R.p("{fig:seq2} shows a backtest. After checking its cache, the service loads the price panel, fits the regime model "
        "on the primary symbol, trains the adaptive policy on the pre-crisis segment only, simulates the three strategies "
        "with the same rebalance dates and cost, and returns values, weights and metrics.")
    R.figure("seq2", FIGD / "fig_seq_backtest.png", "Sequence diagram of a portfolio backtest.", width=5.9)
