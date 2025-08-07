import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(page_title="Sector ETF Analysis", layout="wide")

# Title
st.title("Sector ETF Analysis Dashboard")

# Sidebar
st.sidebar.header("Settings")

# Define sector ETFs
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Financial': 'XLF',
    'Healthcare': 'XLV',
    'Consumer Discretionary': 'XLY',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Materials': 'XLB',
    'Industrial': 'XLI',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE'
}

# Time period selection
period_options = {
    '1 Month': '1mo',
    '3 Months': '3mo',
    '6 Months': '6mo',
    '1 Year': '1y',
    '2 Years': '2y'
}
selected_period = st.sidebar.selectbox('Select Time Period', list(period_options.keys()))

# Risk-free rate input
risk_free_rate = st.sidebar.number_input('Risk-Free Rate (%)', value=2.0, min_value=0.0, max_value=10.0, step=0.1) / 100

@st.cache_data
def fetch_sector_data(etfs, period):
    """Fetch historical data for sector ETFs"""
    # First get SPY data as benchmark
    try:
        spy = yf.download('SPY', period=period_options[period], auto_adjust=False)
        if spy.empty or 'Close' not in spy.columns:
            st.error("Failed to fetch SPY data")
            return pd.DataFrame(), None
        benchmark = spy['Close'].squeeze()  # Ensure 1D series
    except Exception as e:
        st.error(f"Error fetching SPY data: {str(e)}")
        return pd.DataFrame(), None

    # Then get sector data
    sector_data = {}
    for sector, symbol in etfs.items():
        try:
            hist = yf.download(symbol, period=period_options[period], auto_adjust=False)
            if not hist.empty and 'Close' in hist.columns:
                sector_data[sector] = hist['Close'].squeeze()  # Ensure 1D series
        except Exception as e:
            st.error(f"Error fetching data for {sector}: {str(e)}")
    
    # Create DataFrame from sector data
    df = pd.DataFrame(sector_data)
    
    # Ensure both the DataFrame and benchmark have the same index
    if not df.empty and benchmark is not None:
        common_index = df.index.intersection(benchmark.index)
        df = df.loc[common_index]
        benchmark = benchmark.loc[common_index]
        
    return df, benchmark

# Fetch data
df, benchmark_prices = fetch_sector_data(SECTOR_ETFS, selected_period)

if df.empty:
    st.error("Failed to fetch sector data. Please try again.")
    st.stop()

# Calculate metrics
daily_returns = df.pct_change().fillna(0)  # Replace NaN with 0 for first day
daily_returns_clean = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()  # Remove any inf values

# Calculate benchmark returns if available
if benchmark_prices is not None:
    benchmark_returns = benchmark_prices.pct_change().fillna(0)
    benchmark_returns_clean = benchmark_returns.replace([np.inf, -np.inf], np.nan)

# Cumulative returns starting from 1
cumulative_returns = (1 + daily_returns).cumprod()

# Annualized metrics (252 trading days)
annual_return = daily_returns_clean.mean() * 252
volatility = daily_returns_clean.std() * np.sqrt(252)  # Historical Volatility
excess_return = annual_return - risk_free_rate
sharpe_ratio = excess_return / volatility

# Calculate Sortino Ratio with proper handling of negative returns
negative_returns = daily_returns_clean[daily_returns_clean < 0]
downside_std = negative_returns.std() * np.sqrt(252)
# Handle case where there are no negative returns
sortino_ratio = pd.Series(index=df.columns, dtype=float)
for col in df.columns:
    if downside_std[col] == 0:
        sortino_ratio[col] = np.inf if excess_return[col] > 0 else -np.inf
    else:
        sortino_ratio[col] = excess_return[col] / downside_std[col]

correlation = daily_returns_clean.corr()

# Beta and Information Ratio calculations (using SPY as market proxy)
beta = pd.Series(1.0, index=df.columns, dtype=float)
information_ratio = pd.Series(np.nan, index=df.columns, dtype=float)
tracking_error = pd.Series(np.nan, index=df.columns, dtype=float)

if benchmark_prices is not None and len(benchmark_returns_clean) > 1:
    try:
        # Align sector and benchmark returns
        aligned_data = pd.DataFrame({
            'benchmark': benchmark_returns_clean
        }).join(daily_returns_clean, how='inner')
        
        if not aligned_data.empty:
            benchmark_data = aligned_data['benchmark']
            benchmark_var = np.var(benchmark_data)
            benchmark_mean_return = benchmark_data.mean() * 252
            
            for sector in df.columns:
                sector_data = aligned_data[sector]
                
                # Calculate Beta
                if benchmark_var > 0:
                    cov = np.cov(sector_data, benchmark_data)[0,1]
                    beta[sector] = cov / benchmark_var
                
                # Calculate Information Ratio
                excess_returns = sector_data - benchmark_data
                tracking_error[sector] = excess_returns.std() * np.sqrt(252)
                
                if tracking_error[sector] > 0:
                    information_ratio[sector] = (annual_return[sector] - benchmark_mean_return) / tracking_error[sector]
    
    except Exception as e:
        st.warning(f"Risk metrics calculation warning: {str(e)}")
else:
    st.warning("Benchmark data unavailable - using default values for risk metrics")

# Layout
col1, col2 = st.columns(2)

# Performance Chart
with col1:
    st.subheader("Sector Performance Analysis")
    fig = px.line(cumulative_returns, title="Total Return Index (Base = 1.0)")
    fig.update_layout(
        height=400,
        xaxis_title="Trading Date",
        yaxis_title="Total Return Index",
        legend_title="SPDR Sector ETFs",
        template="plotly_dark",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# Volatility Chart
with col2:
    st.subheader("Sector Volatility")
    fig = px.bar(
        x=volatility.index,
        y=volatility.values * 100,  # Convert to percentage
        title="Annualized Volatility",
        template="plotly_dark",
        labels={'x': '', 'y': '%'}
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis_tickformat=',.1f',
        bargap=0.2
    )
    fig.update_traces(
        marker_color='rgb(99, 110, 250)',
        texttemplate='%{y:.1f}%',
        textposition='outside'
    )
    st.plotly_chart(fig, use_container_width=True)

# Correlation Heatmap
st.subheader("Sector Correlation Matrix")

# Create heatmap with custom text annotations
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        z=correlation,
        x=correlation.columns,
        y=correlation.columns,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        showscale=True
    )
)

# Add text annotations
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        fig.add_annotation(
            x=correlation.columns[i],
            y=correlation.columns[j],
            text=f"{correlation.iloc[j, i]:.2f}",
            showarrow=False,
            font=dict(color='white', size=10)
        )

fig.update_layout(
    title="Correlation Matrix",
    height=600,
    template="plotly_dark",
    xaxis_title="",
    yaxis_title="",
    xaxis={'side': 'bottom'},
    yaxis={'side': 'left'}
)

st.plotly_chart(fig, use_container_width=True)

# Risk-Return Analysis
st.subheader("Risk-Return Analysis")
col3, col4 = st.columns(2)

with col3:
    # Scatter plot of risk vs return
    fig = px.scatter(
        x=volatility * 100,  # Convert to percentage
        y=annual_return * 100,  # Convert to percentage
        text=volatility.index,
        title="Risk vs Return Analysis",
        labels={'x': 'Risk (Annualized Volatility, %)', 'y': 'Return (Annualized, %)'},
        template="plotly_dark"
    )
    # Add the Capital Market Line
    market_return = annual_return.mean()
    market_risk = volatility.mean()
    slope = (market_return - risk_free_rate) / market_risk
    x_range = np.linspace(0, (volatility.max() * 100) * 1.1, 100)
    y_range = (risk_free_rate + slope * (x_range/100)) * 100  # Adjust for percentage scale
    fig.add_trace(go.Scatter(
        x=x_range, 
        y=y_range, 
        name='Capital Market Line', 
        line=dict(dash='dash', color='red'),
        hovertemplate='Risk: %{x:.1f}%<br>Return: %{y:.1f}%'
    ))
    
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(
        height=400,
        showlegend=True,
        annotations=[
            dict(
                x=volatility.max(),
                y=risk_free_rate,
                xref="x",
                yref="y",
                text=f"Risk-free rate: {risk_free_rate:.2%}",
                showarrow=False,
                yshift=10
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    # Sharpe Ratio comparison
    fig = px.bar(
        x=sharpe_ratio.index,
        y=sharpe_ratio.values,
        title="Sharpe Ratio by Sector",
        template="plotly_dark",
        labels={'x': '', 'y': 'Sharpe Ratio'}
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis_tickformat='.2f',
        bargap=0.2
    )
    fig.update_traces(
        marker_color='rgb(99, 110, 250)',
        texttemplate='%{y:.2f}',
        textposition='outside'
    )
    st.plotly_chart(fig, use_container_width=True)

# Detailed Statistics
st.subheader("Detailed Statistics")

# Calculate max drawdown for each sector
max_drawdown = pd.Series(index=df.columns)
for column in df.columns:
    prices = df[column]
    peak = prices.expanding(min_periods=1).max()
    drawdown = (prices - peak) / peak
    max_drawdown[column] = drawdown.min()

stats_df = pd.DataFrame({
    'Ann. Return (%)': annual_return * 100,
    'Historical Vol (%)': volatility * 100,
    'Sharpe Ratio': sharpe_ratio,
    'Sortino Ratio': sortino_ratio,
    'Beta': beta,
    'Info Ratio': information_ratio,
    'Max Drawdown (%)': max_drawdown * 100
})

# Format and display statistics with better styling
formatted_stats = stats_df.style\
    .format({
        'Ann. Return (%)': '{:.2f}%',
        'Historical Vol (%)': '{:.2f}%',
        'Sharpe Ratio': '{:.2f}',
        'Sortino Ratio': '{:.2f}',
        'Beta': '{:.2f}',
        'Info Ratio': '{:.2f}',
        'Max Drawdown (%)': '{:.2f}%'
    })\
    .background_gradient(cmap='RdYlGn', subset=['Ann. Return (%)', 'Sharpe Ratio', 'Info Ratio'])\
    .background_gradient(cmap='RdYlGn_r', subset=['Historical Vol (%)', 'Max Drawdown (%)'])

st.dataframe(formatted_stats)
