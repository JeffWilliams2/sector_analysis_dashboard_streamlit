# Sector ETF Analysis Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sector-analysis.streamlit.app)

An interactive financial dashboard for analyzing U.S. sector ETFs, built with Streamlit and Python. The dashboard provides sector analysis, risk metrics, and correlation insights.

## Features

### 1. Interactive Time Period Selection
- 1 Month
- 3 Months
- 6 Months
- 1 Year
- 2 Years

### 2. Key Visualizations
- **Performance Analysis**: Track sector performance with base index of 1.0
- **Volatility Comparison**: Compare annualized volatility across sectors
- **Correlation Matrix**: Interactive heatmap showing sector correlations
- **Sharpe Ratio Comparison**: Compare risk-adjusted returns

### 3. Detailed Statistics Table
- Annualized Returns
- Historical Volatility
- Ratios
- Beta

## Data Sources

The dashboard uses the following Sector ETFs:
- Technology (XLK)
- Financial (XLF)
- Healthcare (XLV)
- Consumer Discretionary (XLY)
- Consumer Staples (XLP)
- Energy (XLE)
- Materials (XLB)
- Industrial (XLI)
- Utilities (XLU)
- Real Estate (XLRE)

Data is fetched real-time using the Yahoo Finance API.


## Requirements

```
streamlit>=1.24.0
pandas>=1.5.0
numpy>=1.23.0
yfinance>=0.2.18
plotly>=5.13.0
```

