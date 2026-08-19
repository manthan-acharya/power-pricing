# Spatiotemporal Power Price Modeling and Stochastic Battery Dispatch

I chose to do this because I thought power pricing was cool lol. The intersection of physics and economics here is fascinating, as physical line limits directly drive dollar prices at every bus.

This repository implements an end-to-end framework for modeling multi-hub electricity price dynamics and solving storage arbitrage as continuous-time stochastic control. The project integrates three core layers:
1. **Physical Grid Mechanics**: Lossless DC optimal power flow (DC-OPF) and Power Transfer Distribution Factor (PTDF) matrix generation on an IEEE 14-bus network to model nodal price divergence under line thermal constraints.
2. **Stochastic Price Dynamics**: A 3-hub neural jump-diffusion model ($X_t \in \mathbb{R}^3$ for NP15, SP15, ZP26 LMPs) incorporating mean reversion toward trailing exponential moving average (EMA) levels, state-dependent Cholesky diffusion, and compound Poisson jumps.
3. **Stochastic Control**: A deep BSDE solver (Han, Jentzen, E, 2018) that solves the Hamilton-Jacobi-Bellman (HJB) equation for a 25 MW / 100 MWh battery asset under viscosity regularization, benchmarked against a grid dynamic program over the same four-dimensional state.

## Repository Structure

- `data/`: Ingestion of 18 months of CAISO day-ahead hourly LMPs (Jan 2025 – Jun 2026) via `gridstatus`, preprocessing, and chronological train/val/test splits.
- `grid/`: DC power flow solver, PTDF matrix calculation, and DC-OPF dual optimization for LMP energy ($\lambda_{\text{sys}}$) and congestion ($C_k$) decomposition.
- `sde/`: Neural jump-diffusion model definition (`model.py`), Euler transition likelihood Stage-A pretraining (`train.py`), Stage-B adjoint-based simulated moment fine-tuning (`torchsde`), block-bootstrap KS tests (`ks_test.py`), and seasonal-OU baselines (`ou_baseline.py`).
- `control/`: Deep BSDE controller (`bsde.py`) solving the forward-backward SDE system for battery dispatch, backward-induction dynamic programming (`baseline_dp.py`) on a reduced single-hub problem, and a full-state grid DP benchmark (`grid_dp.py`) over SOC and all three hubs.
- `backtest/`: Out-of-sample backtesting (`engine.py`) over Q2 2026 realized SP15 prices, moving-block bootstrap confidence intervals, and perfect-foresight LP bounds (`baselines.py`).
- `writeup/`: LaTeX paper (`writeup.tex`) and compiled report (`writeup.pdf`).

## Key Empirical & Policy Results

### Out-of-Sample Dispatch (Q2 2026 Realized Prices)
Evaluated on a 25 MW / 100 MWh battery asset ($\eta = 0.88$, $c_{\text{deg}} = \$2/\text{MWh}$) at SP15 over 2,160 test hours (April–June 2026):

| Dispatch Strategy | Realized Revenue ($) | % of Upper Bound |
| :--- | :---: | :---: |
| **Perfect-Foresight Upper Bound (LP)** | **$259,704** | **100.0%** |
| **Deep BSDE Policy** | **$229,885** | **88.5%** |
| Grid DP Policy (full 4-D state) | $229,511 | 88.4% |
| Hour-of-Day Schedule (5 h charge, 4 h discharge) | $227,170 | 87.5% |
| Rolling Percentile (168-hour window) | $98,468 | 37.9% |
| Static Percentile (Train-set bounds) | $4,722 | 1.8% |

- **No measurable edge over a fixed schedule**: The deep BSDE policy captures 88.5% (95% CI 84.8–92.6%) of perfect foresight. A fixed schedule that charges in the five cheapest and discharges in the four dearest hours of the training split's mean day captures 87.5% (83.4–92.0%). Five and four are the full-power hours needed to fill and to drain the battery. The difference is +1.0 pp (95% CI −2.6 to +4.5 pp, paired t = 0.45), which is not statistically distinguishable from zero. An earlier four-hour charge window could store at most 93.8 MWh and made the policy look 7.0 pp better than it is.
- **Where the policy's revenue comes from**: It charges through the midday solar trough, but 84% of its discharge happens between midnight and 07:00. The $36.5/MWh terminal salvage, set from training-year overnight prices, sits above test-quarter evening prices (mean $25/MWh), so the policy holds energy past midnight.
- **Deep BSDE vs grid DP**: At three hubs, a grid DP over SOC and all three prices is tractable, taking about 3 minutes per day on one core with a sparse truncated kernel. Its policy earns $229,511 against the deep BSDE's $229,885 (paired t = −0.06). The deep BSDE's scaling advantage only matters at four or more price nodes.

### Model Diagnostics & Numerical Validation
- **Seed Robustness**: Across 200 simulation seeds, 73.5% of draws pass all 15 moment checks. Fitted level half-lives (4.14 h) mean-revert faster than empirical observations (13.04 h).
- **DP Cross-Check**: Deep BSDE value function estimates match grid dynamic programming within 15% relative error across 57 of 60 evaluation states. The 3 breaching states occur at low state-of-charge (10 MWh) and short remaining horizons ($t \in \{12, 18\}$ h).

## Pipeline Execution

Install dependencies (`torch`, `torchsde`, `pandapower`, `cvxpy`, `gridstatus`, `scipy`, `matplotlib`) and run single-threaded (`OMP_NUM_THREADS=1`) to prevent BLAS thread exhaustion:

```bash
python3 data/download.py
python3 data/process.py
python3 -m grid.dcopf
python3 -m grid.congestion_link
python3 -m sde.train
python3 -m sde.validate
python3 -m sde.calibration
python3 -m sde.ou_baseline
python3 -m sde.seed_robustness
python3 -m control.bsde
python3 -m control.baseline_dp
python3 -m control.grid_dp
python3 -m backtest.engine
```

To compile the LaTeX writeup:
```bash
cd writeup && pdflatex writeup.tex && pdflatex writeup.tex
```
