# Holdout evaluation protocol (July–September 2026)

Written and committed before any July–September 2026 price was downloaded.
Everything below is fixed. If a bug turns up after the holdout run, it is
fixed and both the original and the corrected numbers are reported.

## Question

Does the stochastic controller beat simple rules that use the same
information, across more than one market regime?

## Periods

| role | window (US/Pacific) | status |
|---|---|---|
| training | 2025-01-01 to 2026-01-31 | used to fit every model; unchanged |
| development | 2026-04-01 to 2026-06-30 | already examined; used to build and debug code only |
| holdout | 2026-07-01 to 2026-09-30 | untouched until the code is frozen; run once |

No strategy parameter is chosen by looking at development-quarter
performance. Holdout results are the primary result. Development results
are reported alongside them as the regime-consistency check.

## Models

- Price model: `sde/checkpoint.pt` as is. No refit on 2026 data.
- Deep BSDE: retrained once, on the same 2025 training days and schedule
  as `control/bsde.py`, with the terminal value `pi` added as a network
  input and sampled uniformly on [0, 60] $/MWh during training. Saved
  alongside the incumbent checkpoint, which is not overwritten.
- 2026 data enters any model only through the causal EMA states at run
  time.

## Terminal value (the one substantive change)

The day-boundary value of stored energy is

    pi_d = eta_d * median(SP15, 00:00-05:59 local, the 7 days before day d) - c_deg

clipped to [0, 60] $/MWh. This is the existing definition with a trailing
week in place of the whole training year. It is applied identically to
every strategy that needs an end-of-day value. The number of clipped days
is reported.

## Strategies

All strategies use the same battery: 25 MW, 100 MWh, sqrt(0.88) each way,
$2/MWh degradation. They start at 50 MWh and carry SOC across days. The
stochastic policies and the forecast baselines all observe the current
hour's price before acting in that hour.

1. **Deep BSDE policy** (retrained as above).
2. **Grid DP policy**: `control/grid_dp.py`, the same grid, with `pi_d`.
3. **Fixed schedule**: charge in the 5 cheapest and discharge in the 4
   dearest hours of the 2025 training split's mean SP15 day.
4. **Forecast MPC, yesterday**: every hour, re-solve the LP over the rest
   of the day. The current hour uses its realized price, later hours use
   the same hour yesterday, and `pi_d` values the end-of-day SOC. Only the
   current hour's action is executed.
5. **Forecast MPC, weekly profile**: as 4, but later hours use the
   trailing 7-day mean price for that hour.

## Ceilings (not strategies)

- Perfect foresight over the whole window, energy revenue only, with no
  salvage term in the objective.
- Perfect foresight solved day by day with `pi_d`, reported for context.

Capture rates are stated against the first.

## Decision rule

The stochastic approach counts as working only if all three hold:

1. On the holdout, the deep BSDE policy beats the best of strategies 3–5.
   The edge is measured in percentage points of the ceiling, and its
   95% moving-block bootstrap CI (5-day blocks, 20,000 resamples) must
   exclude zero.
2. The edge is positive in at least 2 of the 3 holdout months.
3. The edge has the same sign on the development quarter.

The grid DP policy is scored the same way. If it matches or beats the
deep BSDE, the result is that the stochastic control is needed but the
neural solver is not.

Anything else is reported as a null or negative result, in the same
place and at the same prominence as a positive one would have been.

## Regime reporting (descriptive, no decision)

For both quarters, the following are reported for every strategy:

- the edge by month;
- the edge by tercile of daily SP15 price range (max minus min, realized);
- the edge on days with any negative price versus days without.

## Amendment 1 (2026-09-30 20:45 UTC, before any holdout data was downloaded)

Made after the development-quarter run, with the owner's approval.

- Two day-ahead baselines join the simple set:
  6. **Day-ahead plan, yesterday**: at midnight, solve the day's LP against
     yesterday's prices with `pi_d` and execute the plan unchanged.
  7. **Day-ahead plan, weekly profile**: as 6, against the trailing 7-day
     mean price for each hour.

  On the development quarter the hourly re-planning baselines (4, 5) did
  worse than a committed day-ahead plan, so the original set understated
  what a simple rule achieves. Adding them can only make the decision rule
  harder to meet. Decision rule 1 now reads "the best of strategies 3–7".
- The deep BSDE runs as specified. A check found its final-hour marginal
  value, which must equal `pi_d`, reads $7–12/MWh high over the
  development range of `pi_d`. It is not modified after seeing
  development results. The grid DP is the exact solution of the same
  control problem and is scored alongside it.
