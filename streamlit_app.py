import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from math import log, sqrt, exp, pi, erf
from datetime import date, datetime, timedelta
from openai import OpenAI

# ----------------- App config -----------------
st.set_page_config(page_title="Option Suite", layout="wide")
st.title("Option Suite — Multi-calculator Launcher")

# ----------------- Math helpers -----------------
def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def norm_pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)

def bs_price_greeks(S, K, T, r, sigma, option_type):
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            price = max(0.0, S - K)
            delta = 1.0 if S > K else 0.0
        else:
            price = max(0.0, K - S)
            delta = -1.0 if S < K else 0.0
        return {'price': price, 'delta': delta, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        price = S * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)
        delta = norm_cdf(d1)
        theta = (-S * norm_pdf(d1) * sigma / (2 * sqrt(T)) - r * K * exp(-r * T) * norm_cdf(d2))
        rho = K * T * exp(-r * T) * norm_cdf(d2)
    else:
        price = K * exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
        delta = -norm_cdf(-d1)
        theta = (-S * norm_pdf(d1) * sigma / (2 * sqrt(T)) + r * K * exp(-r * T) * norm_cdf(-d2))
        rho = -K * T * exp(-r * T) * norm_cdf(-d2)

    gamma = norm_pdf(d1) / (S * sigma * sqrt(T))
    vega = S * norm_pdf(d1) * sqrt(T)

    return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}

def implied_vol_bisect(market_price, S, K, T, r, option_type, tol=1e-6, max_iter=100):
    if option_type == 'call':
        lower_bound = max(0.0, S - K * exp(-r*T))
    else:
        lower_bound = max(0.0, K * exp(-r*T) - S)
    if market_price <= lower_bound + 1e-12:
        return 1e-12

    low, high = 1e-8, 5.0
    mid = 0.1
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        price = bs_price_greeks(S, K, T, r, mid, option_type)['price']
        if abs(price - market_price) < tol:
            return max(mid, 1e-8)
        if price > market_price:
            high = mid
        else:
            low = mid
    return max(mid, 1e-8)

# ----------------- Navigation bar -----------------
PAGES = {}
def nav_bar_horizontal(selected):
    labels = [meta['label'] for key, meta in PAGES.items()]
    keys = [key for key, meta in PAGES.items()]
    choice = st.radio("", labels, index=labels.index(PAGES[selected]['label']), horizontal=True)
    return keys[labels.index(choice)]

# ----------------- Calculator Page -----------------
def calculator_main():
    st.header("Option Greek Estimator")

    left, right = st.columns([1,2])
    with left:
        st.subheader("Inputs")
        ticker = st.text_input("Ticker (e.g. AAPL)", value="AAPL")
        option_type = st.selectbox("Option type", ['call','put'], index=0)
        option_market_price = st.number_input("Observed option price (per share)", value=1.0, min_value=0.0, format="%.6f")
        strike = st.number_input("Strike price", value=100.0, min_value=0.0, format="%.6f")
        maturity = st.date_input("Maturity / Expiry date", value=date.today())
        r = st.number_input("Risk-free rate (annual, decimal)", value=0.03, format="%.6f")
        calculate = st.button("Calculate")

    with right:
        if not calculate:
            st.info("Change inputs on the left and press **Calculate** to run the estimator.")
            return

        today = date.today()
        if maturity <= today:
            st.error("Maturity must be a future date.")
            return

        try:
            tkr = yf.Ticker(ticker)
            hist = tkr.history(period='5d')
            S = float(hist['Close'].iloc[-1]) if hist is not None and len(hist)>0 else None
        except:
            S = None

        if S is None:
            st.error("Could not fetch underlying price.")
            return

        T = (maturity - today).days / 365.0

        # Try option chain implied vol
        found_iv = None
        try:
            expiry_str = maturity.strftime('%Y-%m-%d')
            if expiry_str in tkr.options:
                chain = tkr.option_chain(expiry_str)
                df = chain.calls if option_type=='call' else chain.puts
                match = df[np.isclose(df['strike'], strike)]
                if len(match)>0 and 'impliedVolatility' in match.columns:
                    iv_val = float(match['impliedVolatility'].iloc[0])
                    if iv_val>5: iv_val/=100.0
                    found_iv = iv_val
        except:
            pass

        if found_iv is not None:
            sigma = found_iv
            st.success(f"Used implied vol from option chain: {sigma:.4f}")
        else:
            sigma = implied_vol_bisect(option_market_price, S, strike, T, r, option_type)
            st.success(f"Solved implied vol from market price: {sigma:.4f}")

        base = bs_price_greeks(S, strike, T, r, sigma, option_type)
        base_price = base['price']

        st.subheader("Calculated Greeks and price")
        greeks_df = pd.DataFrame({
            'Metric': ['Option market price','Model price (BS)','Implied vol','Delta','Gamma','Vega','Theta','Rho'],
            'Value':[option_market_price, round(base_price,6), round(sigma,6), round(base['delta'],6), round(base['gamma'],8), round(base['vega'],6), round(base['theta'],6), round(base['rho'],6)]
        })
        st.table(greeks_df.set_index('Metric'))

        pct_moves = list(range(-20,25,5))
        rows = []
        for pct in pct_moves:
            newS = S*(1+ pct/100)
            new_price = bs_price_greeks(newS, strike, T, r, sigma, option_type)['price']
            abs_change = new_price - base_price
            pct_change = (abs_change/base_price*100) if base_price!=0 else np.nan
            rows.append({'Stock move %': f"{pct}%", 'New underlying': round(newS,4), 'Option price': round(new_price,6), 'Abs change': round(abs_change,6), 'Pct change (%)': round(pct_change,4)})

        moves_df = pd.DataFrame(rows)
        st.subheader("Option price for stock moves (-20% to 20% in 5% steps)")
        st.dataframe(moves_df)
        st.line_chart(moves_df.set_index('Stock move %')['Option price'])

# ----------------- Placeholder Pages -----------------
def stop_loss_calc():
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

def ai_summary():
    # st.header("AI Summary")
    # st.info("Adjust defaults here. No session storage, values reset on refresh.")

    # ------------ CONFIG ------------
    st.set_page_config(page_title="AI Stock Summary", layout="wide")
    
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    
    st.title("📈 AI Stock Summary")
    
    ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT)").upper()
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
    
            # ---------- FUNDAMENTALS ----------
            info = stock.info
    
            company_name = info.get("longName", ticker)
            sector = info.get("sector", "N/A")
            industry = info.get("industry", "N/A")
            market_cap = info.get("marketCap", None)
            pe_ratio = info.get("trailingPE", None)
            eps = info.get("trailingEps", None)
            roe = info.get("returnOnEquity", None)
            debt = info.get("totalDebt", None)
            revenue_growth = info.get("revenueGrowth", None)
    
            # ---------- PRICE DATA ----------
            data = stock.history(period="1y")
            data["EMA_20"] = data["Close"].ewm(span=20).mean()
            data["EMA_50"] = data["Close"].ewm(span=50).mean()
    
            # RSI calculation
            delta = data["Close"].diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean()
            avg_loss = pd.Series(loss).rolling(14).mean()
            rs = avg_gain / avg_loss
            data["RSI"] = 100 - (100 / (1 + rs))
    
            # price trend
            trend = (
                "Uptrend" if data["EMA_20"].iloc[-1] > data["EMA_50"].iloc[-1]
                else "Downtrend"
            )
    
            # ---------- DISPLAY DATA ----------
            st.subheader(f"📌 {company_name} ({ticker})")
    
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Fundamentals")
                st.write(f"**Sector:** {sector}")
                st.write(f"**Industry:** {industry}")
                st.write(f"**Market Cap:** {market_cap:,}" if market_cap else "N/A")
                st.write(f"**P/E Ratio:** {pe_ratio}")
                st.write(f"**EPS:** {eps}")
                st.write(f"**ROE:** {roe}")
                st.write(f"**Revenue Growth:** {revenue_growth}")
                st.write(f"**Total Debt:** {debt}")
    
            with col2:
                st.write("### Technicals")
                st.write(f"**Latest Close:** {round(data['Close'].iloc[-1], 2)}")
                st.write(f"**EMA 20:** {round(data['EMA_20'].iloc[-1], 2)}")
                st.write(f"**EMA 50:** {round(data['EMA_50'].iloc[-1], 2)}")
                st.write(f"**RSI (14):** {round(data['RSI'].iloc[-1], 2)}")
                st.write(f"**Trend:** {trend}")
    
            # ---------- AI SUMMARY ----------
            st.subheader("🤖 AI Summary")
            prompt = f"""
            Provide a clear, concise analyst-style summary for the stock {ticker}.
    
            FUNDAMENTALS:
            - Company: {company_name}
            - Sector: {sector}
            - Industry: {industry}
            - Market Cap: {market_cap}
            - PE: {pe_ratio}
            - EPS: {eps}
            - ROE: {roe}
            - Revenue Growth: {revenue_growth}
            - Total Debt: {debt}
    
            TECHNICALS:
            - Latest Price: {round(data['Close'].iloc[-1], 2)}
            - EMA 20: {round(data['EMA_20'].iloc[-1], 2)}
            - EMA 50: {round(data['EMA_50'].iloc[-1], 2)}
            - RSI: {round(data['RSI'].iloc[-1] if not np.isnan(data['RSI'].iloc[-1]) else 50, 2)}
            - Trend: {trend}
    
            Create:
            - A simple bullish/bearish/neutral view
            - Key risks
            - Short-term technical view
            - Long-term fundamental view
            """
    
            ai_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
    
            st.write(ai_response.choices[0].message["content"])
    
        except Exception as e:
            st.error(f"Error fetching data: {e}")


# ----------------- Register pages -----------------
PAGES['profit_loss']={'label':'Options Profit Loss','func':calculator_main}
PAGES['stop_loss']={'label':'Stop Loss','func':stop_loss_calc}
PAGES['ai_summary']={'label':'AI Summary','func':ai_summary}

# ----------------- Run App -----------------
selected = nav_bar_horizontal('profit_loss')
page_func = PAGES[selected]['func']
page_func()
