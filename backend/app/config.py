"""
Static configuration: default multi-asset universe and the three historical
case studies referenced in the project report (internet/dot-com boom & bust,
2008 Global Financial Crisis, COVID-19 crash).

Dates are chosen so that every ticker in a case study's asset list actually
has trading data for (most of) the requested window -- Yahoo Finance ETF
histories are shorter than the underlying events for some instruments (e.g.
GLD only starts trading in Nov 2004), so the dot-com case study swaps in
longer-history futures/instruments instead of ETFs.
"""

DEFAULT_ASSETS = ["SPY", "QQQ", "GLD", "TLT"]

# A rebalance every ~21 trading days (roughly monthly) for every strategy,
# so equal-weight / mean-variance / adaptive are compared on equal footing.
REBALANCE_FREQ_DAYS = 21

# Round-trip transaction cost assumption applied to portfolio turnover at
# each rebalance (5 basis points), per the report's "incorporate transaction
# costs into the reward function" research gap.
TRANSACTION_COST_BPS = 5

CASE_STUDIES = {
    "dotcom": {
        "id": "dotcom",
        "label": "Dot-com / Internet Boom & Bust",
        "start": "1999-06-01",
        "end": "2003-12-31",
        "crisis_start": "2000-03-10",
        "crisis_end": "2002-10-09",
        "assets": ["SPY", "QQQ", "GC=F"],
        # The event's most DISTINCTIVE, headline move is QQQ's ~78% collapse
        # (vs. SPY's milder ~49%) -- used as the symbol for the historical
        # shape-comparison corpus (event_matcher.py) so the corpus and any
        # query against it are built from the same, most-recognizable shape.
        "shape_symbol": "QQQ",
        "description": (
            "The late-1990s internet boom drove tech valuations to extremes; "
            "the Nasdaq (proxied here by QQQ) peaked on 10 Mar 2000 and fell "
            "~78% by Oct 2002. TLT/GLD ETFs don't exist yet for this period, "
            "so gold futures (GC=F) stand in as the defensive asset."
        ),
    },
    "gfc": {
        "id": "gfc",
        "label": "2008 Global Financial Crisis",
        "start": "2006-01-01",
        "end": "2010-12-31",
        "crisis_start": "2007-10-09",
        "crisis_end": "2009-03-09",
        "assets": ["SPY", "QQQ", "GLD", "TLT"],
        "description": (
            "The subprime mortgage collapse triggered a systemic banking "
            "crisis; the S&P 500 fell ~57% peak-to-trough between Oct 2007 "
            "and Mar 2009."
        ),
    },
    "covid": {
        "id": "covid",
        "label": "COVID-19 Crash & Recovery",
        "start": "2019-01-01",
        "end": "2021-12-31",
        "crisis_start": "2020-02-19",
        "crisis_end": "2020-04-07",
        "assets": ["SPY", "QQQ", "GLD", "TLT"],
        "description": (
            "Global lockdowns triggered the fastest ~34% bear-market decline "
            "in S&P 500 history (19 Feb - 23 Mar 2020), followed by an "
            "unusually sharp, liquidity-driven recovery."
        ),
    },
}

# Symbols shown in the live market ribbon on the dashboard.
LIVE_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "BTC-USD", "^VIX"]

RISK_FREE_RATE_ANNUAL = 0.02
TRADING_DAYS_PER_YEAR = 252
