import streamlit as st
import pandas as pd

st.set_page_config(page_title="LEAPS Stop-Loss Calculator", layout="wide")

st.title("📉 Stock Option Stop-Loss Calculator (Greek Adjusted)")
st.write("Automatically calculate stop-loss for LEAPS using premium, Greeks, and intelligent % rules.")

# --- Inputs ---
st.header("Input Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    entry_price = st.number_input(
        "Entry Premium",
        min_value=0.01,
        value=1.40,
        step=0.01,
        help="The price you paid for the option contract (Ask price for BUY orders)."
    )

with col2:
    stop_loss_percent = st.number_input(
        "Stop-Loss %",
        min_value=1,
        max_value=90,
        value=30,
        step=1,
        help="The percentage drop from your entry price at which the stop-loss should trigger."
    )

with col3:
    days_to_hold = st.number_input(
        "Days Expected Before Review",
        min_value=1,
        value=30,
        help="How long you plan to hold the LEAPS before reviewing the position. Used for Theta-based decay calculations."
    )

col4, col5 = st.columns(2)
with col4:
    theta = st.number_input(
        "Theta (daily)",
        value=-0.002,
        step=0.001,
        format="%.4f",
        help="Daily time decay of the option premium. Typically a small negative number for LEAPS."
    )

with col5:
    delta = st.number_input(
        "Delta",
        value=0.20,
        step=0.01,
        help="Sensitivity of the option price to a $1 move in the underlying. Higher delta = stronger correlation."
    )

st.divider()

# --- Raw Stop-Loss ---
raw_sl = entry_price * (1 - stop_loss_percent / 100)

# --- Theta Decay ---
theta_decay = abs(theta) * days_to_hold

# --- Greek-adjusted stop loss ---
greek_adjusted_sl = entry_price - ((entry_price * stop_loss_percent / 100) - theta_decay)

# --- Results ---
st.header("Results")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Raw Stop-Loss (No Greeks)", f"{raw_sl:.2f}")

with colB:
    st.metric("Theta Decay Over Period", f"{theta_decay:.2f}")

with colC:
    st.metric("Greek-Adjusted Stop-Loss", f"{greek_adjusted_sl:.2f}")

# --- Table for multiple SL levels ---
st.subheader("Stop-Loss Levels Summary")

levels = [20, 25, 30, 35, 40]

data = []
for lvl in levels:
    raw = entry_price * (1 - lvl / 100)
    adjusted = entry_price - ((entry_price * lvl / 100) - theta_decay)
    data.append((lvl, round(raw, 2), round(adjusted, 2)))

df = pd.DataFrame(data, columns=["SL %", "Raw SL", "Greek Adjusted SL"])

st.dataframe(df, width='stretch')