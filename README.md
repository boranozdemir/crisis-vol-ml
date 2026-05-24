# Crisis Volatility Forecasting with GARCH and Machine Learning

## Overview

This project studies volatility dynamics during the 2008 mortgage crisis using financial econometrics and machine learning methods. The main focus is on whether negative market shocks had a stronger effect on future volatility during the crisis, especially in the U.S. financial and banking sectors.

The project combines classical GARCH-family models with modern machine learning benchmarks. GARCH and GJR-GARCH models are used to capture volatility clustering and asymmetric shock effects, while machine learning models are used as flexible forecasting benchmarks for one-step-ahead volatility prediction.

The empirical analysis is based on daily market data for broad market and financial sector assets around the 2008 crisis period.

---

## Research Question

The main research question is:

> Did the 2008 mortgage crisis strengthen the asymmetric impact of negative returns on future volatility in the U.S. financial sector?

A secondary research question is:

> Can machine learning models improve one-step-ahead volatility forecasts relative to GARCH-family models during crisis and post-crisis periods?

The project does not attempt to forecast the direction of asset prices. Instead, it focuses on conditional volatility and risk.

---

## Motivation

Financial crises are periods in which volatility becomes highly persistent and market reactions to negative news may become stronger. During the 2008 mortgage crisis, financial institutions, banks, and real estate-related assets were directly exposed to severe stress. This makes the crisis period a natural setting for studying asymmetric volatility.

A standard GARCH model captures volatility clustering, but it treats positive and negative shocks symmetrically. In financial markets, this assumption is often restrictive because negative returns may increase future volatility more than positive returns of the same magnitude.

The GJR-GARCH model addresses this issue by allowing negative shocks to have an additional effect on conditional variance. In this project, the asymmetry parameter is interpreted as a measure of crisis-related downside risk sensitivity.

Machine learning models are added as forecasting benchmarks. Unlike GARCH models, they do not impose a specific parametric volatility structure. Instead, they learn the relationship between lagged return-based features and future realized volatility proxies.

---

## Data

The planned asset universe includes broad market, financial sector, and banking sector proxies.

Main assets:

```text
SPY  - S&P 500 ETF
XLF  - Financial Select Sector ETF
KBE  - Bank ETF