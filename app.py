import datetime
import math

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================================================
# 1. PAGE CONFIG & THEME (GREEN – SUSTAINABILITY FOCUS)
# =========================================================
st.set_page_config(
    page_title="PeakGuard AI – Smart Green Building Forecaster",
    page_icon="🌱",
    layout="wide",
)


def inject_green_theme():
    """Inject a simple green, modern theme via CSS."""
    st.markdown(
        """
        <style>
        /* Page background */
        .stApp {
            background: radial-gradient(circle at top left, #0f5132 0, #020b07 55%, #000000 100%);
            color: #f5f5f5;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #052e25 0%, #02130e 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.06);
        }

        /* Cards look for metrics */
        div[data-testid="stMetric"] {
            background: rgba(0, 0, 0, 0.35);
            border-radius: 12px;
            padding: 12px;
            border: 1px solid rgba(0, 255, 153, 0.35);
        }

        /* Headers */
        h1, h2, h3, h4 {
            color: #e9fff5 !important;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSelectbox div[data-baseweb="select"],
        .stDateInput input,
        .stTimeInput input {
            background-color: rgba(3, 25, 20, 0.9) !important;
            color: #e9fff5 !important;
            border-radius: 8px !important;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #22c55e, #4ade80);
            color: #02110b;
            border-radius: 999px;
            border: none;
            padding: 0.45rem 1.4rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            filter: brightness(1.05);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.6);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_green_theme()

# =========================================================
# 2. LOAD MODEL + FEATURES
# =========================================================


@st.cache_resource(show_spinner=False)
def load_model_and_files():
    """
    Load trained LightGBM model and feature list.
    Expects:
        - lgbm_model.txt
        - feature_names.pkl
    """
    model = lgb.Booster(model_file="lgbm_model.txt")
    model_features = joblib.load("feature_names.pkl")
    all_primary_uses = [
        col.replace("primary_use_", "")
        for col in model_features
        if col.startswith("primary_use_")
    ]
    return model, model_features, all_primary_uses


model, model_features, all_primary_uses = load_model_and_files()


# =========================================================
# 3. FEATURE ENGINEERING
# =========================================================
def create_features_from_input(
    air_temp: float,
    hour: int,
    day_of_week: int,
    month: int,
    primary_use: str,
    square_feet: float,
    year_built: int,
    lag1: float,
    lag24: float,
):
    """
    Build a single-row dataframe matching the model's training features.
    """
    input_data = {
        "air_temperature": air_temp,
        "square_feet": square_feet,
        "year_built": year_built,
        "month": month,
        "meter_reading_lag1": lag1,
        "meter_reading_lag24": lag24,
    }

    # Heating / Cooling degree days
    input_data["hdd"] = max(0.0, 18.0 - air_temp)
    input_data["cdd"] = max(0.0, air_temp - 21.0)

    # Time cyclic encodings
    input_data["hour_sin"] = math.sin(2 * math.pi * hour / 24.0)
    input_data["hour_cos"] = math.cos(2 * math.pi * hour / 24.0)
    input_data["day_of_week_sin"] = math.sin(2 * math.pi * day_of_week / 7.0)
    input_data["day_of_week_cos"] = math.cos(2 * math.pi * day_of_week / 7.0)

    # One-hot primary_use
    for use in all_primary_uses:
        input_data[f"primary_use_{use}"] = 1 if use == primary_use else 0

    # Any missing features from training get 0 by default
    df = pd.DataFrame([input_data])
    df = df.reindex(columns=model_features, fill_value=0)
    return df


# =========================================================
# 4. WEATHER API (REAL-TIME)
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_weather_for_now(latitude: float, longitude: float):
    """
    Fetch current-hour temperature using Open-Meteo (no API key required).
    Returns (air_temp_c, hour, day_of_week, month)
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m",
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=6)
        resp.raise_for_status()
        data = resp.json()

        times = data["hourly"]["time"]
        temps = data["hourly"]["temperature_2m"]

        # Take the first hour >= now in the API's timezone
        now = datetime.datetime.now()
        # Times are ISO strings like "2025-11-23T14:00"
        best_idx = 0
        best_diff = None
        for i, t in enumerate(times):
            ts = datetime.datetime.fromisoformat(t)
            diff = abs((ts - now).total_seconds())
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_idx = i

        ts = datetime.datetime.fromisoformat(times[best_idx])
        temp_c = float(temps[best_idx])

        hour = ts.hour
        day_of_week = ts.weekday()  # Monday=0
        month = ts.month
        return temp_c, hour, day_of_week, month
    except Exception as e:
        # Fail gracefully – caller will show a warning
        return None, None, None, None


# =========================================================
# 5. SOLAR GENERATION + CO₂ LOGIC
# =========================================================
def estimate_solar_generation_kwh(solar_capacity_kw: float, hour: int) -> float:
    """
    Very simple static solar profile:
    - 0 kWh outside [6, 18]
    - Peaks around 12–13
    """
    if solar_capacity_kw <= 0:
        return 0.0

    if hour < 6 or hour > 18:
        return 0.0

    # Triangular shape peaking at noon (12:00)
    peak_hour = 12
    span = 6  # from 6 to 18
    factor = max(0.0, 1.0 - abs(hour - peak_hour) / span)
    # Assume each kW produces ~factor kWh in that hour (oversimplified but fine for demo)
    return solar_capacity_kw * factor


def compute_co2(energy_kwh: float, emission_factor_kg_per_kwh: float) -> float:
    return max(0.0, energy_kwh) * max(0.0, emission_factor_kg_per_kwh)


# =========================================================
# 6. SIDEBAR – BUILDING, CONTRACT, SUSTAINABILITY
# =========================================================
with st.sidebar:
    st.markdown("### 🌱 PeakGuard AI – Setup")

    st.markdown("**1. Building profile**")
    primary_use = st.selectbox("Primary building use", options=all_primary_uses)
    square_feet = st.number_input(
        "Building size (sq. ft.)", min_value=1_000, max_value=1_000_000, value=50_000, step=1_000
    )
    year_built = st.number_input(
        "Year built", min_value=1900, max_value=datetime.datetime.now().year, value=2005, step=1
    )

    st.markdown("---")
    st.markdown("**2. Contract & emissions**")
    contract_peak_limit = st.number_input(
        "Contract Peak Limit (kW)",
        min_value=0.0,
        max_value=10000.0,
        value=500.0,
        step=10.0,
        help="Your utility's contracted maximum demand. For 1-hour intervals, kW ≈ kWh.",
    )
    emission_factor = st.number_input(
        "Grid emission factor (kg CO₂ per kWh)",
        min_value=0.0,
        max_value=2.0,
        value=0.82,  # approx India grid
        step=0.01,
        help="Ask sustainability/utility team if you want an exact value.",
    )

    st.markdown("---")
    st.markdown("**3. Solar integration**")
    solar_capacity_kw = st.number_input(
        "On-site solar capacity (kW)",
        min_value=0.0,
        max_value=5000.0,
        value=0.0,
        step=10.0,
        help="Set to 0 if the building has no solar.",
    )

    st.markdown("---")
    st.markdown("**4. Live weather (optional)**")
    use_live_weather = st.checkbox("Use real-time weather API", value=False)
    latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=23.0, step=0.1)
    longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=72.0, step=0.1)

    if use_live_weather:
        if st.button("Fetch current weather"):
            temp_c, hour_api, dow_api, month_api = fetch_weather_for_now(latitude, longitude)
            if temp_c is None:
                st.error("Couldn't fetch weather. Check your internet connection.")
            else:
                # Store in session_state so the main form can pick it up
                st.session_state["live_air_temp"] = temp_c
                st.session_state["live_hour"] = hour_api
                st.session_state["live_day_of_week"] = dow_api
                st.session_state["live_month"] = month_api
                st.success(
                    f"Weather loaded – {temp_c:.1f} °C at {hour_api:02d}:00, "
                    f"Day {dow_api} (0=Mon) Month {month_api}"
                )

    st.markdown("---")
    st.caption("Tip: Use live weather + solar to showcase *real* green impact in your demo.")


# =========================================================
# 7. MAIN LAYOUT – TABS
# =========================================================
st.title("🌱 PeakGuard AI – Smart Green Building Forecaster")

tab_forecast, tab_howto = st.tabs(["⚡ Forecast & Scenarios", "📘 How to use App"])


# ---------------------------------------------------------
# TAB 1 – FORECAST & SCENARIOS
# ---------------------------------------------------------
with tab_forecast:
    st.subheader("Set operating conditions")

    # Default to "now" but allow manual override
    today = datetime.date.today()
    col_date, col_time = st.columns(2)
    with col_date:
        date_selected = st.date_input("Date", value=today)
    with col_time:
        default_time = datetime.time(
            hour=st.session_state.get("live_hour", 14), minute=0
        )
        time_selected = st.time_input("Hour of day", value=default_time, step=3600)

    # Convert to model inputs
    hour = time_selected.hour
    day_of_week = date_selected.weekday()  # Monday=0
    month = date_selected.month

    col_temp, col_lag1, col_lag24 = st.columns(3)

    with col_temp:
        default_temp = st.session_state.get("live_air_temp", 25.0)
        air_temp = st.number_input(
            "Outdoor air temperature (°C)",
            min_value=-30.0,
            max_value=50.0,
            value=float(default_temp),
            step=0.5,
        )

    with col_lag1:
        lag1 = st.number_input(
            "Energy 1 hour ago (kWh)",
            min_value=0.0,
            max_value=20000.0,
            value=100.0,
            step=10.0,
            help="Use last hour's meter reading.",
        )

    with col_lag24:
        lag24 = st.number_input(
            "Energy 24 hours ago (kWh)",
            min_value=0.0,
            max_value=20000.0,
            value=110.0,
            step=10.0,
            help="Same hour, previous day.",
        )

    st.markdown("")

    run_btn = st.button("🔮 Run forecast & scenario analysis")

    if run_btn:
        # ---------------- PREDICTION ----------------
        features_df = create_features_from_input(
            air_temp=air_temp,
            hour=hour,
            day_of_week=day_of_week,
            month=month,
            primary_use=primary_use,
            square_feet=square_feet,
            year_built=year_built,
            lag1=lag1,
            lag24=lag24,
        )

        pred_log = model.predict(features_df)[0]
        pred_kwh = float(np.expm1(pred_log))  # model predicted log(1 + kWh)

        # For 1-hour window, kW ≈ kWh
        predicted_demand_kw = pred_kwh

        # ---------------- SOLAR + CO₂ ----------------
        solar_gen_kwh = estimate_solar_generation_kwh(solar_capacity_kw, hour)

        grid_kwh_no_solar = pred_kwh
        grid_kwh_with_solar = max(0.0, pred_kwh - solar_gen_kwh)

        co2_no_solar = compute_co2(grid_kwh_no_solar, emission_factor)
        co2_with_solar = compute_co2(grid_kwh_with_solar, emission_factor)
        co2_savings = co2_no_solar - co2_with_solar

        # ---------------- CONTRACT PEAK STATUS ----------------
        if contract_peak_limit > 0:
            margin_kw = contract_peak_limit - predicted_demand_kw
            if margin_kw >= 50:
                peak_status = "SAFE"
                peak_color = "✅"
                peak_msg = "Comfortable margin below contract peak."
            elif 0 <= margin_kw < 50:
                peak_status = "AT RISK"
                peak_color = "🟡"
                peak_msg = "Close to contract limit – consider pre-cooling or shifting loads."
            else:
                peak_status = "BREACH"
                peak_color = "🚨"
                peak_msg = "Forecast exceeds contract peak – high demand charges likely."
        else:
            peak_status = "N/A"
            peak_color = "ℹ️"
            peak_msg = "No contract peak limit specified."

        # ---------------- METRICS ROW ----------------
        st.markdown("### Results")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Predicted consumption (next hour)",
            f"{pred_kwh:,.2f} kWh",
        )

        if contract_peak_limit > 0:
            delta_kw = predicted_demand_kw - contract_peak_limit
            m2.metric(
                f"Contract peak status – {peak_color} {peak_status}",
                f"{predicted_demand_kw:,.1f} kW",
                f"{delta_kw:+.1f} kW vs limit",
            )
        else:
            m2.metric("Contract peak status", "No limit set")

        m3.metric(
            "Solar generation (this hour, est.)",
            f"{solar_gen_kwh:,.2f} kWh",
        )

        st.markdown("---")

        # ---------------- CO2 + SCENARIO SUMMARY ----------------
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Grid energy – without solar",
            f"{grid_kwh_no_solar:,.2f} kWh",
        )
        c2.metric(
            "Grid energy – with solar",
            f"{grid_kwh_with_solar:,.2f} kWh",
            f"{grid_kwh_with_solar - grid_kwh_no_solar:+.2f} kWh",
        )
        c3.metric(
            "CO₂ avoided this hour",
            f"{co2_savings:,.2f} kg",
        )

        # Short narrative for judges
        st.markdown(
            f"""
            **What this scenario shows:**

            - For **{date_selected} at {hour:02d}:00**, PeakGuard AI forecasts **~{pred_kwh:,.0f} kWh**.
            - With your contract limit of **{contract_peak_limit:,.0f} kW**, status is **{peak_color} {peak_status}** – {peak_msg}
            - With **{solar_capacity_kw:,.0f} kW** solar, the model estimates **{solar_gen_kwh:,.1f} kWh** on-site generation this hour.
            - That cuts grid draw from **{grid_kwh_no_solar:,.1f} → {grid_kwh_with_solar:,.1f} kWh**, avoiding **~{co2_savings:,.1f} kg CO₂** in just one hour.
            """
        )


# ---------------------------------------------------------
# TAB 2 – HOW TO USE (FOR YOU + JUDGES)
# ---------------------------------------------------------
with tab_howto:
    st.markdown(
        """
        ### How to operate the app

        **Step 1 – Set the building context (left sidebar)**  
        - Choose *Primary building use* (Office, Education, Healthcare, etc.)  
        - Enter floor area and year built.  
        - Set **Contract Peak Limit (kW)** from your utility bill.  
        - Add **solar capacity** if the building has rooftop PV.  
        - Set a realistic **grid emission factor** (0.7–0.9 for India is typical).

        **Step 2 – Pull live conditions (optional, but impressive)**  
        - Turn on **“Use real-time weather API”**.  
        - Enter latitude & longitude for the campus or building.  
        - Click **“Fetch current weather”** – this will auto-fill temperature & time.

        **Step 3 – Configure the operating scenario**  
        - In the *Forecast & Scenarios* tab:
          - Adjust **Date** and **Hour** you want to simulate (e.g., 15:00 today).  
          - Provide **Energy 1 hour ago** and **24 hours ago** from the meter or from a realistic assumption.

        **Step 4 – Run the AI forecast**  
        - Click **“Run forecast & scenario analysis”**.  
        - The app will:
          1. Engineer all time + weather features.
          2. Use your trained LightGBM model to predict next-hour consumption.
          3. Compare it with your **contract peak**.
          4. Estimate **solar generation** and **CO₂ savings**.

        **How this makes you stand out in competition:**
        - Not just forecasting – you are **connecting AI → cost → CO₂ → solar** in one screen.  
        - You show **“What if?” scenarios** for:
          - Different times of day  
          - With/without solar  
          - Different contract peak limits  

        Structure your pitch like this:
        1. *“PeakGuard AI predicts dangerous peaks **before** they happen.”*
        2. *“We integrate solar and emissions so facility managers see both **money** and **carbon** impact.”*
        3. *“This is plug-and-play for any commercial building with smart meters and basic weather data.”*
        """
    )
