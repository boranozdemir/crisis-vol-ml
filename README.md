# Crisis Volatility Forecasting with GARCH

## Overview
This project investigates volatility dynamics during the 2008 Subprime Mortgage Crisis using rigorous financial econometrics. The objective is to evaluate how extreme market shocks impact future volatility across a hierarchical market structure: the broad equity market, the financial sector, and the banking sector epicenter. We compare symmetric GARCH models with asymmetric specifications (GJR-GARCH, EGARCH) using a 1-step ahead expanding-window out-of-sample (OOS) forecasting framework.

## Research Questions
1. **The Leverage Effect:** Did the 2008 crisis amplify the asymmetric impact of negative returns on future volatility across different market levels?
2. **Parsimony vs. Complexity:** Do complex asymmetric models consistently outperform simpler, symmetric models during chaotic market stress, or does parsimony offer better robustness?

## Data
The asset universe tracks volatility contagion from the macro-economy down to the micro-level epicenter:
* **SPY (Macro):** SPDR S&P 500 ETF (Broad U.S. Equity Market)
* **XLF (Meso):** Financial Select Sector SPDR (Financial Sector)
* **^BKX (Micro/Epicenter):** KBW Nasdaq Bank Index (Banking Sector). *Utilized instead of recent ETFs to eliminate small-sample bias during pre-crisis estimation.*

## Methodology & Models
The econometric workflow includes estimating models with Student-t innovations to account for heavy-tailed financial returns:
* GARCH(1,1), GARCH(1,2), GARCH(2,1)
* GJR-GARCH(1,1)
* EGARCH(1,1)

**Key analytical steps:**
1. Pre-estimation diagnostics (ADF, Jarque-Bera, ARCH-LM).
2. Model optimization and parameter estimation.
3. Post-estimation diagnostics (Ljung-Box, residual ARCH-LM, Engle-Ng Sign Bias).
4. 1-step ahead expanding-window OOS volatility forecasting.

## Evaluation Metrics
Forecast accuracy is strictly audited using:
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **QLIKE** (Quasi-Likelihood Loss)

**QLIKE** is emphasized as the primary ranking metric because it heavily penalizes the dangerous underestimation of volatility, acting as an essential safeguard against model risk during financial meltdowns.