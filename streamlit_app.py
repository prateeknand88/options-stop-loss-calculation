import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Option Stop-Loss Calculator")
st.markdown("This tool calculates stop-loss levels for stock options.")

# --- User Inputs ---
premium = st.number_input("Option Premium ($)", min_value=0.0, step=0.01, help="Enter the current option premium")
sl_percent = st.number_input("Stop-Loss Percentage (%)", min_value=1.0, max_value=99.0, value=30.0, help="Percentage drop from premium at which stop-loss should trigger")

# --- Advanced Inputs ---
st.markdown("---")
st.subheader("Advanced: Fetch Option Chain")
use_advanced = st.checkbox("Fetch from API (yfinance)")

if use_advanced:
    ticker = st.text_input("Stock Ticker", help="Enter the underlying US stock ticker, e.g., AAPL, TSLA")
    strike_price = st.number_input("Strike Price", min_value=0.0, step=0.5, help="Strike price of the option")
    expiry = st.text_input("Expiration Date (YYYY-MM-DD)", help="Option expiration date")
    option_type = st.radio("Option Type", ["Call", "Put"], help="Select Call or Put")

    if st.button("Fetch & Calculate Stop Loss"):
        try:
            # Fetch option chain from yfinance
            ticker_obj = yf.Ticker(ticker)
            opt_chain = ticker_obj.option_chain(expiry)
            if option_type.lower() == 'call':
                option_data = opt_chain.calls
            else:
                option_data = opt_chain.puts

            option_row = option_data[option_data['strike'] == strike_price]

            if option_row.empty:
                st.error("No option data found for this strike and expiry.")
            else:
                premium_api = float(option_row['lastPrice'].values[0])
                sl_level_api = premium_api * (1 - sl_percent / 100)

                st.subheader("Results from API")
                st.write(f"**Current Premium:** ${premium_api:.2f}")
                st.write(f"**Stop-Loss Premium ({sl_percent}%):** ${sl_level_api:.2f}")

                # Suggested Stop-Loss Table
                percentages = list(range(10, 60, 10))
                levels = [premium_api * (1 - p/100) for p in percentages]
                df_sl = pd.DataFrame({"Stop-Loss %": percentages, "Premium Level ($)": [f"{x:.2f}" for x in levels]})
                st.table(df_sl)

                # # Display option row with Greeks
                # st.subheader("Option Data with Greeks")
                # display_cols = ['contractSymbol', 'strike', 'lastPrice', 'bid', 'ask', 'change', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'delta', 'gamma', 'theta', 'vega', 'rho']
                # # Some columns may not exist, filter those that exist
                # available_cols = [col for col in display_cols if col in option_row.columns]
                # st.table(option_row[available_cols])

        except Exception as e:
            st.error(f"Error fetching option chain from yfinance: {e}")

# --- Manual Calculation ---
else:
    if st.button("Calculate Stop Loss"):
        sl_level = premium * (1 - sl_percent / 100)

        st.subheader("Manual Results")
        st.write(f"**Current Premium:** ${premium:.2f}")
        st.write(f"**Stop-Loss Premium ({sl_percent}%):** ${sl_level:.2f}")

        st.subheader("Suggested Stop-Loss Levels")
        percentages = list(range(10, 60, 10))
        levels = [premium * (1 - p/100) for p in percentages]
        df = pd.DataFrame({"Stop-Loss %": percentages, "Premium Level ($)": [f"{x:.2f}" for x in levels]})
        st.table(df)