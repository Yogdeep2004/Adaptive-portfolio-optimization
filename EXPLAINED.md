# So, what does this thing actually do?

*A walkthrough of the Adaptive Portfolio Optimizer — starting in plain English,
ending in the actual code.*

---

## Part 1: Explain it like I'm not a coder

Imagine you have some money spread across a few different investments — say,
some in big U.S. stocks, some in tech stocks, some in gold, some in bonds.
Most of the time, that mix just quietly goes up and down a little, and it's
fine to leave it alone.

But every so often, something big happens. The dot-com bubble bursts. A
housing crisis freezes the banks. A pandemic shuts the world down overnight.
And when that happens, "just leave it alone" can be a really expensive
decision.

This app tries to help with two very human questions:

1. **"Wait — does this remind you of something?"**
   If I show you a chart of the last few weeks or months, can a computer
   recognize "hey, this looks a lot like the start of 2008" or "this looks
   like nothing bad — this is just a normal, boring stretch of the market"?

2. **"If a robot were managing this money, would it actually do better?"**
   Instead of always keeping the same fixed mix of investments, what if the
   computer watched for warning signs and quietly shifted more into safer
   assets when trouble seemed to be brewing — and back into growth assets
   once things calmed down? Would that actually beat just leaving it alone?

That's it. That's the whole project. Everything else is just *how* it
answers those two questions.

### The three things you can actually click on

- **Portfolio Backtest** — pick one of three famous crashes (the dot-com
  bust, the 2008 financial crisis, or the COVID crash), and watch three
  different "investing strategies" race against each other through that
  exact period: one that never adjusts, one that adjusts using classic
  1950s-era math, and one that adjusts using the "watch for warning signs"
  robot. You get to see, in real numbers, whether the robot actually helped.

- **Regime Detection** — pick any stock and any date range, and the app will
  (a) color-code each day as calm, shaky, or full-blown crisis, and (b) tell
  you, in plain language, whether that stretch of time resembles one of the
  three historical crashes — or whether it's just a normal period, in which
  case it gives you a plain statistical "here's what usually happens next"
  instead.

- **About / Tech Stack** — the nerdy version of this document, inside the
  app itself.

Okay. Now let's slowly turn up the technical detail.

---

## Part 2: The ideas, without the code yet

Before we get to library names and file paths, here are the four *ideas*
this app is built on — explained with analogies first.

### Idea 1 — "Teach a computer to notice a mood shift, without telling it what to look for"

If you asked a friend "was 2008 a scary year for the stock market?", they'd
say yes immediately — but nobody sat them down and gave them a legal
definition of "crisis." They just absorbed, from experience, that big
volatile drops *feel* different from calm periods.

We do something similar, but statistically. Every day, the market has a
"mood": how much did it move, how volatile has it been lately, how far is it
below its recent peak. We feed years of those daily moods into a model and
say: *"Without me telling you what a crash looks like, figure out that
there seem to be roughly three different 'moods' the market falls into —
and tell me, for every single day, which mood it was probably in."*
The model discovers "calm," "declining," and "panicky" entirely on its own.

### Idea 2 — "Compare a ripple to a tsunami by squishing them to the same size"

Here's a genuinely tricky problem: the COVID crash took about a month.
The dot-com crash took about two and a half *years*. How do you compare
the "shape" of a one-month event to the shape of a multi-year event?

The trick: stretch or squeeze every time period onto the same ruler. Take
whatever window you're looking at — 20 days or 2,000 — and describe it as
"40 checkpoints, evenly spaced from the start of the period to the end."
Now a lightning-fast crash and a slow-motion grind can be laid side by side
and compared as *shapes*, regardless of how long each one actually took.

### Idea 3 — "A robot that learns by trial and error, like training a dog"

For the "should the robot adjust the portfolio" strategy, we don't hand-code
rules like "if volatility > X, sell stocks." Instead, we let a small
decision-making "brain" try thousands of slightly different strategies on
historical data, see which ones would have produced good outcomes (decent
returns, without huge losses, without flipping the portfolio around too
often and racking up trading fees), and nudge the brain toward whatever
worked. Repeat that a few dozen times, and the brain gets noticeably better
— the same basic idea as clicker-training a dog, just done with math instead
of treats.

### Idea 4 — "Sort similar things into piles, without being told what the piles are"

Once we can describe any time period as a "shape," we take *thousands* of
historical shapes — some from the dot-com era, some from 2008, some from
COVID, some from perfectly ordinary years — and ask an algorithm to group
them into piles of similar shapes, with zero hints about which pile is
which. It's only *after* the piles form that we peek at which historical
events ended up in each pile, and label that pile accordingly. Critically,
this kind of grouping is also allowed to say "this doesn't belong in any
pile" — which is exactly how the app decides "this doesn't look like any of
the three famous crashes."

Those four ideas — mood detection, shape-squishing, trial-and-error
learning, and unsupervised grouping — are the whole intellectual core of
this project. Now let's see exactly which real tools implement them.

---

## Part 3: The actual tech stack, and where each idea lives in it

### The skeleton: a backend and a frontend

Like almost any modern app, this is two programs talking to each other:

- **The backend** (`backend/`) is written in Python using **FastAPI**. It
  does all the actual thinking — fetching prices, running the models,
  crunching the numbers — and exposes that as a set of URLs (an API) that
  return JSON.
- **The frontend** (`frontend/`) is written in **React**, built with
  **Vite**, and draws all the charts with a library called **Recharts**. It
  never does any of the "thinking" itself — it just asks the backend
  questions and draws pretty pictures of the answers.

They run as two separate local servers and talk over plain HTTP, which is
why `run.sh` starts both of them at once.

### Where the prices come from

Real, live prices come from **Yahoo Finance's public price data feed**
(`backend/app/data/market_data.py`). One fun implementation detail: this
code deliberately does *not* use Python's popular `requests` library to
fetch that data — it shells out to `curl` instead. During development, Yahoo
started silently rate-limiting `requests`-style traffic far more
aggressively than plain `curl`, even from the exact same network, for
reasons that are basically "bot-fingerprinting." Switching the HTTP client
to `curl` fixed it instantly. It's a small, slightly funny reminder that
"free public APIs" often come with unwritten rules. As a safety net, if
Yahoo is ever completely unreachable, the app quietly swaps in a clearly
labeled, statistically-calibrated fake dataset for the three historical
case studies, so a live demo never just breaks.

### Idea 1, for real: `hmmlearn`'s Gaussian Hidden Markov Model

The "mood detector" is a **Hidden Markov Model** (`backend/app/ml/regime.py`),
built with the `hmmlearn` library. It's fed three numbers per day — that
day's return, a 20-day rolling volatility, and a 60-day drawdown-from-peak —
and told to find **3 hidden states**. Because the model has no idea which
state is "calm" and which is "panic" (it just calls them state 0, 1, 2), the
code re-labels them afterward by their own statistics: whichever state has
the highest average volatility becomes **Crisis**, and of the remaining two,
the one with the better average return becomes **Bull**, leaving **Bear**.
This model is completely unsupervised — it is never told the real dates of
any crash — and it correctly lights up bright red exactly on the COVID crash
when you try it, which is a satisfying thing to watch happen.

### Idea 2, for real: a Transformer that turns a "shape" into a list of 16 numbers

This is the newest piece (`backend/app/ml/embedding.py`). Any price window —
short or long — first gets converted into three tracks: cumulative return,
short-term rolling volatility, and running drawdown, all measured *from that
window's own starting point*. Then, regardless of how many actual trading
days were in the original window, it gets resampled down to exactly **40
evenly spaced checkpoints** (that's Idea 2's "squish it to the same ruler,"
made literal).

Those 40 checkpoints get fed into a small **Transformer encoder** — yes, the
same family of architecture behind large language models, just tiny here:
2 layers, 4 attention heads, an internal width of 32 numbers, compressing
down to a **16-number "fingerprint"** of the shape. It's trained the way you
train an autoencoder: force it to compress a shape down to 16 numbers and
then reconstruct the original 40-checkpoint shape from just those 16, and
punish it for reconstruction errors. Do that over and over on a training set
of real historical windows, and the 16 numbers it settles on end up encoding
something meaningful about the shape — not because anyone told it what
"meaningful" means, but because compressing-and-reconstructing forces it to.

To be upfront about scale: this is trained on roughly **900 historical
windows**, not the billions of examples a real large model sees. What it
actually learns is closer to a smart, nonlinear compression of the
return/volatility/drawdown path than deep "understanding" — but that's
still a genuine, honestly-scoped use of self-attention, and importantly it's
one that a simple average or a handful of summary statistics *can't* do:
it can represent not just *how bad* a window was, but *where in the window*
the damage happened.

### Idea 4, for real: scikit-learn's HDBSCAN

Once every historical window (and your selected window) has been reduced to
16 numbers, `backend/app/ml/event_matcher.py` builds the reference corpus:
thousands of overlapping windows of many different lengths, sliced out of
the three real historical crises plus several long, boring, calm reference
periods, each one labeled by how much it actually overlaps that crisis's
real date range. (A subtlety worth mentioning: COVID's actual crash lasted
only about 33 trading days, so — left alone — it would produce far fewer
labeled examples than the multi-year dot-com and 2008 declines. The code
explicitly caps and balances how many windows of each label and each length
get kept, so the rare, fast COVID pattern doesn't get statistically drowned
out by the slow ones.)

**`sklearn.cluster.HDBSCAN`** then groups all those 16-number fingerprints
into clusters, with zero hints about which cluster is which — it just finds
where the fingerprints are dense. *After* clustering, each group gets
labeled by whatever historical event its members mostly belong to, but only
if that majority is above **60% purity** — otherwise, that whole cluster
just means "normal." Your selected window gets compared against the nearest
cluster's center, and only counts as a real "match" if it's close enough,
confident enough, and that cluster isn't the "normal" one. If none of that
holds, the app is honest about it and reports "no strong resemblance,"
falling back to a plain statistical baseline instead of forcing a fake
answer.

### Idea 3, for real: a tiny neural network trained with Evolution Strategies

For the "adaptive" trading strategy (`backend/app/ml/allocator.py`), the
"brain" is a small 2-layer network that looks at the current mix of
regime-mood probabilities plus each asset's recent momentum and volatility,
and outputs how much of the portfolio to put into each asset.

It's trained with **Evolution Strategies** — spin up roughly **32 random
variations** of the brain's internal settings, simulate each one trading
through history, and nudge the "average" brain a little bit toward whichever
variations did best (good risk-adjusted return, small drawdowns, not too
much needless buying-and-selling). Repeat that about **40 times**. This is a
real, published reinforcement-learning technique — it just doesn't use the
GPU-hungry, autodiff-heavy training loop that more famous RL algorithms
(like PPO, which the actual research papers behind this project use) rely
on, which is exactly why it can train in a handful of seconds on an ordinary
laptop instead of needing serious hardware.

Crucially, the brain is only ever *trained* on data from *before* a crisis
starts, and then evaluated walking forward through the real crash — so
whatever advantage it shows isn't just memorization; it's genuinely reacting
to a pattern it's never seen play out before.

### How a click actually turns into an answer

When you open the Regime Detection tab, pick a symbol and dates, and hit
Analyze, roughly this happens:

1. The frontend fires off two separate requests to the backend at once —
   one for the day-by-day mood coloring, one for the historical comparison
   — deliberately kept independent so a slow comparison never blanks out
   the chart above it.
2. The backend fetches real prices, runs the Hidden Markov Model, and sends
   back a color-coded timeline.
3. In parallel, it turns your selected window into its 40-checkpoint shape,
   runs it through the trained Transformer to get its 16-number fingerprint,
   and checks which HDBSCAN cluster it's closest to.
4. It writes a short, honest, numbers-backed report — this isn't a canned
   sentence, every figure in it (confidence %, drawdown, volatility, "what
   usually happens next") is computed fresh from your actual selection —
   and sends it back.
5. The frontend draws your period's shape and a real historical reference
   window side by side, so you can *see*, not just be told, whether the
   trend lines actually match.

That's the whole system: one classic statistical model doing mood-detection,
one modern neural architecture doing shape-recognition, one unsupervised
clustering algorithm doing pattern-matching, and one small
trial-and-error-trained network doing the actual portfolio decisions — all
wired together behind a couple of buttons.
