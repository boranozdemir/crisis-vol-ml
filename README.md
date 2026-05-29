# Volatility Forecasting During the 2008 Global Financial Crisis

## Overview

This study examines volatility dynamics in the U.S. equity market during the 2008 Global Financial Crisis. The analysis focuses on three assets: the SPDR S&P 500 ETF (SPY), representing the broad U.S. market; the Financial Select Sector SPDR Fund (XLF), representing the financial sector; and the KBW Nasdaq Bank Index (BKX), representing the banking sector.

The project mainly uses GARCH-family models to study how volatility changes over time, how persistent volatility becomes during the crisis, and whether negative returns have a stronger effect on future volatility. Student-t innovations are used because financial returns often exhibit heavy tails.

In addition to the econometric models, machine learning models such as Random Forest and XGBoost are included as forecasting benchmarks. The project also considers a simple hybrid structure, where GARCH-based volatility forecasts are used as input features for machine learning models. In this way, the machine learning models do not rely only on lagged returns, but also learn from the volatility signal produced by the econometric models.

Overall, the project combines traditional volatility modeling, machine learning benchmarks, and a hybrid forecasting structure to evaluate how different approaches perform during a highly unstable market period.