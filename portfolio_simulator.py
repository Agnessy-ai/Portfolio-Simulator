# Streamlit Monte Carlo Portfolio Simulator
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Step 1: App Title
st.title("\U0001F4C8 Monte Carlo Portfolio Simulator")

# Step 2: Inputs
tickers_input = st.text_input("Enter stock tickers (comma-separated)", "AAPL, TSLA, MSFT")
tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

weights_input = st.text_input("Enter corresponding weights (comma-separated)", "0.4, 0.3, 0.3")

try:
    weights = [float(w.strip()) for w in weights_input.split(',') if w.strip() and w.strip() != '.']
    if len(weights) != len(tickers):
        st.error("⚠️ The number of weights must match the number of tickers.")
        st.stop()
    if not np.isclose(sum(weights), 1.0):
        st.warning(f"Weights don't sum to 1 (currently {sum(weights):.2f}). Normalizing them.")
        weights = [w / sum(weights) for w in weights]
except ValueError:
    st.error("Please enter valid numeric weights separated by commas.")
    st.stop()

initial_investment = st.number_input("Initial investment amount ($)", min_value=1000, value=10000, step=100)
years = st.slider("Years to forecast", 1, 30, 10)
num_simulations = st.slider("Number of simulations", 100, 10000, 1000, step=100)
adjust_inflation = st.checkbox("Adjust for 3% annual inflation", value=True)

# Step 3: Download Historical Data
@st.cache_data
def get_data(tickers):
    try:
        data = yf.download(tickers, period='5y', group_by='ticker', auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):  # Multiple tickers
            adj_close = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers if 'Adj Close' in data[ticker]})
        else:  # Single ticker
            adj_close = pd.DataFrame(data['Adj Close'])
            adj_close.columns = [tickers[0]]
        return adj_close
    except Exception as e:
        st.error(f"❌ Failed to download data: {e}")
        return pd.DataFrame()

prices = get_data(tickers)

# Step 4: Validate and Clean Data
if prices.empty or prices.isnull().values.any():
    st.error("❌ Price data contains missing values or failed to load. Check tickers.")
    st.stop()

valid_columns = prices.columns[prices.notna().all()]
invalid_columns = set(tickers) - set(valid_columns)

if invalid_columns:
    st.warning(f"⚠️ The following tickers have missing data and will be excluded: {', '.join(invalid_columns)}")
    prices = prices[valid_columns]
    weights = weights[:len(valid_columns)]

if prices.shape[1] < 2:
    st.error("❌ Not enough valid tickers with historical data.")
    st.stop()

# Step 5: Returns and Covariance
daily_returns = prices.pct_change().dropna()
mean_returns = daily_returns.mean() * 252
cov_matrix = daily_returns.cov() * 252
cov_matrix += np.eye(len(cov_matrix)) * 1e-6  # Regularize

# Step 6: Monte Carlo Simulation with Monthly Rebalancing and Optional Inflation Adjustment
def monte_carlo_simulation(mean_returns, cov_matrix, weights, initial_investment, years, num_simulations, adjust_inflation):
    results = []
    all_returns = []
    months = years * 12
    monthly_returns = mean_returns / 12
    monthly_cov = cov_matrix / 12
    inflation_rate = 0.03

    for _ in range(num_simulations):
        simulated_returns = np.random.multivariate_normal(monthly_returns, monthly_cov, months)
        portfolio_value = initial_investment
        weights_array = np.array(weights)
        run_returns = []

        for r_vector in simulated_returns:
            asset_values = weights_array * portfolio_value
            asset_values *= (1 + r_vector)
            portfolio_value = np.sum(asset_values)
            weights_array = np.array(weights)  # Reset monthly
            run_returns.append(np.dot(r_vector, weights))

        if adjust_inflation:
            portfolio_value /= ((1 + inflation_rate) ** years)

        results.append(portfolio_value)
        all_returns.append(run_returns)

    return results, all_returns

# Step 7: Run Simulation
if st.button("Run Simulation"):
    final_values, all_returns = monte_carlo_simulation(
        mean_returns, cov_matrix, weights, initial_investment, years, num_simulations, adjust_inflation
    )

    flat_returns = np.concatenate(all_returns)
    sharpe_ratio = (np.mean(flat_returns) / np.std(flat_returns)) * np.sqrt(12)  # Monthly Sharpe

    fig, ax = plt.subplots()
    ax.hist(final_values, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(initial_investment, color='red', linestyle='--', label='Initial Investment')
    ax.set_title('Monte Carlo Simulation Results')
    ax.set_xlabel('Portfolio Value ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

    # Metrics
    st.subheader("\U0001F4CC Key Stats")
    st.write(f"**Mean final portfolio value:** ${np.mean(final_values):,.2f}")
    st.write(f"**Median final portfolio value:** ${np.median(final_values):,.2f}")
    st.write(f"**5th percentile (worst case):** ${np.percentile(final_values, 5):,.2f}")
    st.write(f"**95th percentile (best case):** ${np.percentile(final_values, 95):,.2f}")
    st.write(f"**Sharpe Ratio (monthly, approx.):** {sharpe_ratio:.2f}")
    st.write(f"**Probability of losing money:** {np.mean(np.array(final_values) < initial_investment) * 100:.2f}%")

    results_df = pd.DataFrame({"Final Portfolio Value": final_values})
    st.download_button("Download Results", results_df.to_csv(index=False), file_name="monte_carlo_simulations.csv")
 