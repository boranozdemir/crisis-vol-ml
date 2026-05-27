from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/data_overview")

# crisis period for shading the charts
CRISIS_START = "2007-07-01"
CRISIS_END = "2009-06-30"

def add_crisis_shade(ax):
    """Adds a gray shaded region to highlight the 2008 mortgage crisis."""
    ax.axvspan(pd.Timestamp(CRISIS_START), pd.Timestamp(CRISIS_END), 
               color='gray', alpha=0.2, label='Crisis Period (2007-2009)')

def create_visualizations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load required data
    prices = pd.read_csv(RAW_DIR / "prices.csv", index_col="Date", parse_dates=True)
    returns = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    vol_proxy = pd.read_csv(PROCESSED_DIR / "volatility_proxy.csv", index_col="Date", parse_dates=True)

    # Set a clean, academic plotting style
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Adjusted Close Prices (All Assets Together)
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    for col in prices.columns:
        ax.plot(prices.index, prices[col], linewidth=1.5, label=col)
    
    add_crisis_shade(ax)
    ax.set_title("Adjusted Close Prices (SPY, XLF, BKX)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_prices_overview.png", dpi=300)
    plt.close()

    # 2. Daily Log Returns (Subplots for comparison)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, asset in enumerate(returns.columns):
        axes[i].plot(returns.index, returns[asset], color=sns.color_palette()[i], linewidth=0.8)
        add_crisis_shade(axes[i])
        axes[i].set_title(f"Daily Log Returns: {asset}")
        axes[i].set_ylabel("Return")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_returns_clustering.png", dpi=300)
    plt.close()

    # 3. Squared Returns / Volatility Proxy (Subplots)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i, asset in enumerate(vol_proxy.columns):
        axes[i].plot(vol_proxy.index, vol_proxy[asset], color=sns.color_palette("dark")[i], linewidth=0.8)
        add_crisis_shade(axes[i])
        axes[i].set_title(f"Volatility Proxy (Squared Returns): {asset}")
        axes[i].set_ylabel("Squared Return")
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_volatility_proxy.png", dpi=300)
    plt.close()

    # 4. Return Distributions (Histograms showing heavy tails)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, asset in enumerate(returns.columns):
        data = returns[asset].dropna()
        
        # Plot the empirical histogram and KDE (Kernel Density Estimate)
        sns.histplot(data, bins=60, stat='density', kde=True, ax=axes[i], 
                     color=sns.color_palette()[i], alpha=0.5, label='Empirical KDE')
        
        # Calculate mean and standard deviation for the Normal Overlay
        mu, std = data.mean(), data.std()
        
        # Create x-axis values for the normal curve spanning the limits of the plot
        xmin, xmax = axes[i].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        
        # Calculate the theoretical normal PDF (Probability Density Function)
        p = norm.pdf(x, mu, std)
        
        # Plot the Normal curve overlay
        axes[i].plot(x, p, 'k', linewidth=2, linestyle='--', label='Normal Dist.')
        
        axes[i].set_title(f"Distribution: {asset}")
        axes[i].set_xlabel("Return")
        axes[i].set_ylabel("Density")
        
        # Add legend to distinguish Empirical vs Normal
        axes[i].legend(loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_return_distributions.png", dpi=300)
    plt.close()

    # 5. Return Correlation Matrix (Heatmap)
    # Proves the strong linear relationship between the broader market and the financial sectors.
    plt.figure(figsize=(8, 6))
    corr_matrix = returns.corr()
    
    # Using coolwarm palette for financial correlations (blue=low/neg, red=high/pos)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8})
    
    plt.title("Return Correlation Matrix (SPY, XLF, BKX)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_return_correlation.png", dpi=300)
    plt.close()

    # 6. 22-Day Rolling Volatility (Annualized)
    # Smooths out the noise of squared returns to clearly show the crisis peak.
    # 22 days roughly equals one trading month.
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Calculate 22-day rolling standard deviation and annualize it (sqrt(252))
    for col in returns.columns:
        rolling_vol = returns[col].rolling(window=22).std() * np.sqrt(252)
        ax.plot(returns.index, rolling_vol, linewidth=1.2, label=col)
    
    add_crisis_shade(ax)
    ax.set_title("22-Day Rolling Volatility (Annualized Realized Volatility)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Annualized Volatility")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_rolling_volatility_22d.png", dpi=300)
    plt.close()

    print(f"Data visualizations successfully saved to: {OUTPUT_DIR}")
    print("Files created: 01_prices_overview, 02_returns_clustering, 03_volatility_proxy, 04_return_distributions, 05_return_correlation, 06_rolling_volatility_22d")

if __name__ == "__main__":
    create_visualizations()