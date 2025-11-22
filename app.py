# app.py  -- PeakGuard AI (Green theme, upgraded)

import datetime as dt
import requests

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------
# 1. PAGE CONFIG + THEME
# ---------------------------------------------------------
st.set_page_config(
    page_title="PeakGuard AI – Smart Green Building Forecaster",
    page_icon="🌿",
    layout="wide",
)

# Custom dark-green theme
st.markdown(
    """
    <style>
        /* Global */
        .stApp {
            background: radial-gradient(circle at top left, #123726 0, #020608 40%, #020304 100%);
            color: #f4f7f5;
            font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #061611;
            border-right: 1px solid rgba(120, 255, 180, 0.25);
        }
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            color: #e5fff1;
        }

        /* Inputs */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stDateInput > div > div > input,
        .stTimeInput > div > div > input {
            background-color: #071710 !important;
            color: #f4f7f5 !important;
            border-radius: 0.4rem !important;
            border: 1px solid rgba(140, 255, 200, 0.2) !important;
        }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #18a34a, #16c47f);
            color: #04110b;
            border-radius: 999px;
            padding: 0.6rem 1.8rem;
            font-weight: 600;
            border: none;
            box-shadow: 0 0 0 1px rgba(0,0,0,0.3), 0 10px 25px rgba(0, 255, 150, 0.25);
        }
        .stButton > button:hover {
            filter: brightness(1.08);
            box-shadow: 0 0 0 1px rgba(0,0,0,0.4), 0 12px 30px rgba(0, 255, 180, 0.3);
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background: rgba(6, 28, 19, 0.9);
            padding: 0.9rem 1.1rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(72, 214, 134, 0.45);
        }

        /* Section titles */
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .section-subtitle {
            font-size: 0.9rem;
            opacity: 0.75;
        }

        /* Tiny badges */
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            padding: 0.12rem 0.65rem;
            border-radius: 999px;
            font-size: 0.65rem;
            letter-spacing: .04em;
            text-transform: uppercase;
        }
        .badge-peak {
            background: rgba(255, 99, 132, 0.14);
            color: #ff9ba8;
        }
        .badge-green {
            background: rgba(72, 214, 134, 0.18);
            color: #8bffd1;
        }

        /* Cards */
        .glass-card {
            background: radial-gradient(circle at top left, rgba(40, 120, 93, 0.65), rgba(5,12,10,0.9));
            border-radius: 1.05rem;
            padding: 1.2rem 1.4rem;
            border: 1px solid rgba(120, 255, 190, 0.35);
            box-shadow: 0 18px 50px rgba(0,0,0,0.55);
        }

        .sub-card {
            background: rgba(3,14,9,0.8);
            border-radius: 0.9rem;
            padding: 0.9rem 1.0rem;
            border: 1px solid rgba(95, 220, 160, 0.35);
        }

        .ai-rec {
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .ai-rec li {
            margin-bottom: 0.25rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 2. LOAD MODEL + FEATURES
# ---------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    model = lgb.Booster(model_file="lgbm_model.txt")
    feature_names = joblib.load("feature_names.pkl")
    # infer valid primary_use categories from feature names
    primary_uses = [
        col.replace("primary_use_", "")
        for col in feature_names
        if col.startswith("primary_use_")
    ]
    return model, feature_names, primary_uses


model, MODEL_FEATURES, ALL_PRIMARY_USES = load_model_and_metadata()


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING – must match training logic
# ---------------------------------------------------------
def build_feature_row(
    air_temp: float,
    hour: int,
    day_of_week: int,
    month: int,
    primary_use: str,
    square_feet: float,
    year_built: int,
    lag1: float,
    lag24: float,
) -> pd.DataFrame:
    """Create a single feature row compatible with the LightGBM model."""

    data = {
        "air_temperature": float(air_temp),
        "square_feet": float(square_feet),
        "year_built": int(year_built),
        "month": int(month),
        "meter_reading_lag1": float(lag1),
        "meter_reading_lag24": float(lag24),
    }

    # Heating / Cooling degree-days
    data["hdd"] = max(0.0, 18.0 - air_temp)   # base 18°C
    data["cdd"] = max(0.0, air_temp - 21.0)   # base 21°C

    # Time encodings – cyclical
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    data["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    data["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    # One-hot primary_use
    for use in ALL_PRIMARY_USES:
        data[f"primary_use_{use}"] = 1 if use == primary_use else 0

    df = pd.DataFrame([data])
    # Reindex to exact model feature order
    df = df.reindex(columns=MODEL_FEATURES, fill_value=0)
    return df


# ---------------------------------------------------------
# 4. WEATHER + SOLAR HELPERS
# ---------------------------------------------------------
def fetch_live_temperature(lat: float, lon: float, when: dt.datetime) -> float | None:
    """Open-Meteo API – returns outdoor temp in °C for given datetime, or None."""
    try:
        base_url = "https://api.open-meteo.com/v1/forecast"
        date_str = when.date().isoformat()
        resp = requests.get(
            base_url,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m",
                "start_date": date_str,
                "end_date": date_str,
                "timezone": "auto",
            },
            timeout=6,
        )
        resp.raise_for_status()
        data = resp.json()
        temps = data.get("hourly", {}).get("temperature_2m")
        times = data.get("hourly", {}).get("time")
        if not temps or not times:
            return None
        target_iso = when.replace(minute=0, second=0, microsecond=0).isoformat(timespec="hours")
        # times from API are "YYYY-MM-DDTHH:00"
        try:
            idx = times.index(target_iso)
            return float(temps[idx])
        except ValueError:
            return None
    except Exception:
        return None


def estimate_solar_generation_kw(
    onsite_capacity_kw: float, hour: int
) -> float:
    """
    Rough solar profile:
    - 0 output at night
    - triangular peak around 12:00
    - this is intentionally simple for demo visualization.
    """
    if onsite_capacity_kw <= 0:
        return 0.0

    if hour < 6 or hour > 18:
        return 0.0

    # hour 6 -> factor 0; 12 -> 1; 18 -> 0 (triangle)
    factor = max(0.0, 1.0 - abs(hour - 12) / 6.0)
    return onsite_capacity_kw * factor  # kWh for 1-hour interval


def classify_peak_status(pred_kwh: float, contract_limit_kw: float) -> tuple[str, float]:
    """
    Compare predicted consumption vs contract limit.
    Returns (status_label, delta).
    delta > 0  -> amount ABOVE limit
    delta <= 0 -> remaining headroom
    """
    if contract_limit_kw <= 0:
        return "No limit set", 0.0

    delta = pred_kwh - contract_limit_kw
    if delta > 0:
        return "BREACH – forecast exceeds contract peak", delta
    elif abs(delta) < 1e-6:
        return "AT LIMIT – exactly at contract peak", 0.0
    else:
        return "SAFE – below contract peak", delta


def generate_ai_recommendations(
    forecast_kwh: float,
    contract_limit_kw: float,
    solar_kw: float,
    grid_emission_factor: float,
    grid_without_solar: float,
    grid_with_solar: float,
    co2_avoided_kg: float,
) -> list[str]:
    recs: list[str] = []

    # Contract status
    if contract_limit_kw > 0:
        diff = forecast_kwh - contract_limit_kw
        pct_over = (diff / contract_limit_kw) * 100 if contract_limit_kw else 0

        if diff > 0:
            recs.append(
                f"⚠️ Forecast is **{diff:.1f} kW above** contract peak. "
                f"Shift non-critical loads (EV charging, pumps, chilled water) "
                f"out of this hour or pre-cool by 1–2°C in the previous hours."
            )
            if pct_over > 10:
                recs.append(
                    "📈 Consider **renegotiating a higher contract peak** for this season "
                    "or adding on-site storage (battery / thermal) to shave peaks."
                )
        elif diff > -0.1 * contract_limit_kw:
            recs.append(
                f"🟡 You’re within **10% of contract**. Put the building on a "
                "temporary ‘demand guard’ mode (tighten set-points, pause non-essential loads)."
            )
        else:
            recs.append(
                "✅ You are **comfortably below** contract peak. This is a good slot for "
                "running energy-intensive tasks."
            )

    # Solar and CO₂
    if solar_kw > 0:
        if co2_avoided_kg > 0.1:
            recs.append(
                f"🌞 With **{solar_kw:.0f} kW** solar, you avoid about **{co2_avoided_kg:.1f} kg CO₂** in this hour. "
                "Highlight this as a concrete sustainability win in your demo."
            )
        else:
            recs.append(
                "🌞 Solar is configured but not helping much in this hour "
                "(probably low load or low irradiance). Show another scenario "
                "where solar cuts a visible chunk of grid draw."
            )
    else:
        recs.append(
            "🌱 Try setting a non-zero **on-site solar capacity (kW)** and re-run. "
            "That instantly demonstrates how renewables shave peaks and reduce CO₂."
        )

    # Emission factor sanity check
    if grid_emission_factor <= 0.1:
        recs.append(
            "ℹ️ Your grid emission factor looks very low. For Indian grids, "
            "a value in the **0.7–0.9 kg CO₂/kWh** range is more realistic – "
            "double-check the number before presenting."
        )

    # Generic scenario suggestion
    recs.append(
        "🧪 In the demo, run **two back-to-back scenarios**: "
        "first with today’s settings, then reduce contract peak or increase solar. "
        "Let the cards + chart show how grid draw and CO₂ respond."
    )

    return recs

# ---------------------------------------------------------
# 5. SIDEBAR – SETUP
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌱 PeakGuard AI – Setup")

    # 1. Building profile
    st.markdown("#### 1. Building profile")
    primary_use = st.selectbox(
        "Primary building use",
        options=ALL_PRIMARY_USES,
        index=ALL_PRIMARY_USES.index("Education") if "Education" in ALL_PRIMARY_USES else 0,
        key="sb_primary_use",
    )

    square_feet = st.number_input(
        "Building size (sq. ft.)",
        min_value=1000,
        max_value=1_000_000,
        value=50_000,
        step=1_000,
        key="sb_sqft",
    )

    year_built = st.number_input(
        "Year built",
        min_value=1900,
        max_value=dt.date.today().year,
        value=2005,
        step=1,
        key="sb_year_built",
    )

    st.markdown("---")

    # 2. Contract & emissions
    st.markdown("#### 2. Contract & emissions")

    contract_peak_limit_kw = st.number_input(
        "Contract Peak Limit (kW)",
        min_value=0.0,
        max_value=100000.0,
        value=500.0,
        step=10.0,
        key="sb_contract_limit",
    )

    grid_emission_factor = st.number_input(
        "Grid emission factor (kg CO₂ per kWh)",
        min_value=0.0,
        max_value=5.0,
        value=0.82,
        step=0.01,
        key="sb_emission_factor",
        help="Approximate kg of CO₂ emitted by the grid per kWh consumed.",
    )

    st.markdown("---")

    # 3. Solar integration
    st.markdown("#### 3. Solar integration")

    onsite_solar_kw = st.number_input(
        "On-site solar capacity (kW)",
        min_value=0.0,
        max_value=100000.0,
        value=0.0,
        step=10.0,
        key="sb_solar_capacity",
    )

    st.caption("Tip: Set a non-zero solar capacity to showcase peak shaving "
               "and CO₂ avoidance in your Sustainathon demo.")

    st.markdown("---")

    # 4. Live weather (optional)
    st.markdown("#### 4. Live weather (optional)")
    use_live_weather = st.checkbox(
        "Use real-time weather API",
        value=False,
        key="sb_use_live_weather",
    )

    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=23.0,
        step=0.1,
        key="sb_latitude",
    )
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=72.0,
        step=0.1,
        key="sb_longitude",
    )

    st.caption(
        "Use live weather + solar to showcase **real** green impact in your demo."
    )

# ---------------------------------------------------------
# 6. MAIN LAYOUT – INPUTS
# ---------------------------------------------------------
st.markdown(
    "<div class='glass-card'>"
    "<div style='display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;'>"
    "<span style='font-size:1.6rem;'>🌿</span>"
    "<div><div style='font-size:1.4rem;font-weight:650;'>PeakGuard AI – Smart Green Building Forecaster</div>"
    "<div style='font-size:0.9rem;opacity:0.8;'>Forecast peaks, test solar & quantify CO₂ savings in seconds.</div></div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown("## ⚡ Forecast & scenarios")

with st.container():
    st.markdown(
        "<div class='section-title'>Set operating conditions</div>"
        "<div class='section-subtitle'>Choose the hour you care about and how the building is behaving right now.</div>",
        unsafe_allow_html=True,
    )

    with st.form("forecast_form"):
        col_date, col_time = st.columns([2, 1])
        with col_date:
            target_date = st.date_input(
                "Date",
                value=dt.date.today(),
                key="in_date",
            )
        with col_time:
            target_time = st.time_input(
                "Hour of day",
                value=dt.time(hour=14, minute=0),
                key="in_time",
            )

        col_temp, col_lag1, col_lag24 = st.columns(3)

        manual_air_temp = col_temp.number_input(
            "Outdoor air temperature (°C)",
            min_value=-30.0,
            max_value=50.0,
            value=25.0,
            step=0.5,
            key="in_air_temp_manual",
        )

        lag1 = col_lag1.number_input(
            "Energy 1 hour ago (kWh)",
            min_value=0.0,
            max_value=100_000.0,
            value=850.0,
            step=10.0,
            key="in_lag1",
        )

        lag24 = col_lag24.number_input(
            "Energy 24 hours ago (kWh)",
            min_value=0.0,
            max_value=100_000.0,
            value=800.0,
            step=10.0,
            key="in_lag24",
        )

        run_btn = st.form_submit_button("Run forecast & scenario analysis")

# ---------------------------------------------------------
# 7. RUN PREDICTION
# ---------------------------------------------------------
if run_btn:
    # Combine date + time
    target_dt = dt.datetime.combine(target_date, target_time)
    hour = target_dt.hour
    day_of_week = target_dt.weekday()  # 0=Mon ... 6=Sun
    month = target_dt.month

    # Live weather override if enabled
    if use_live_weather:
        live_temp = fetch_live_temperature(latitude, longitude, target_dt)
        if live_temp is not None:
            air_temp = live_temp
            st.info(
                f"🌤️ Using live weather: {air_temp:.1f} °C from Open-Meteo for "
                f"{target_dt.strftime('%Y-%m-%d %H:00')} at ({latitude:.2f}, {longitude:.2f})."
            )
        else:
            air_temp = manual_air_temp
            st.warning(
                "Live weather API call failed – falling back to manual temperature input.",
                icon="⚠️",
            )
    else:
        air_temp = manual_air_temp

    # Build feature row and predict
    feature_row = build_feature_row(
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

    log_pred = model.predict(feature_row)
    pred_kwh = float(np.expm1(log_pred[0]))

    # Scenario calculations
    solar_kwh = estimate_solar_generation_kw(onsite_solar_kw, hour)
    grid_without_solar_kwh = pred_kwh
    grid_with_solar_kwh = max(0.0, pred_kwh - solar_kwh)

    co2_without_solar = grid_without_solar_kwh * grid_emission_factor
    co2_with_solar = grid_with_solar_kwh * grid_emission_factor
    co2_avoided = co2_without_solar - co2_with_solar

    peak_status, delta_kw = classify_peak_status(pred_kwh, contract_peak_limit_kw)

    # -----------------------------------------------------
    # RESULTS – TOP METRICS
    # -----------------------------------------------------
    st.markdown("## Results")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(
            label="Predicted consumption (next hour)",
            value=f"{pred_kwh:,.2f} kWh",
        )
    with m2:
        status_delta = (
            f"+{delta_kw:,.1f} kW vs limit" if delta_kw > 0 else f"{abs(delta_kw):,.1f} kW headroom"
        )
        st.metric(
            label=f"Contract peak status – {'🚨' if 'BREACH' in peak_status else '✅'} {peak_status}",
            value=f"{pred_kwh:,.1f} kW",
            delta=status_delta if contract_peak_limit_kw > 0 else None,
        )
    with m3:
        st.metric(
            label="Solar generation (this hour, est.)",
            value=f"{solar_kwh:,.2f} kWh",
        )

    # -----------------------------------------------------
    # GRID VS SOLAR + CO2
    # -----------------------------------------------------
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Grid energy – without solar")
        st.markdown(
            f"<div class='sub-card'><span style='font-size:1.6rem;font-weight:650;'>{grid_without_solar_kwh:,.2f} kWh</span>"
            f"<div style='font-size:0.85rem;opacity:0.8;'>Baseline forecast if the site had no solar.</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### Grid energy – with solar")
        delta_grid = grid_with_solar_kwh - grid_without_solar_kwh
        st.markdown(
            f"<div class='sub-card'><span style='font-size:1.6rem;font-weight:650;'>{grid_with_solar_kwh:,.2f} kWh</span>"
            f"<div style='font-size:0.85rem;opacity:0.8;'>After applying on-site solar for this hour.</div>"
            f"<div style='margin-top:0.3rem;font-size:0.8rem;color:#8bffd1;'>"
            f"{'↓' if delta_grid < 0 else '→'} {abs(delta_grid):,.2f} kWh vs. no-solar baseline\"</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown("#### CO₂ avoided this hour")
        st.markdown(
            f"<div class='sub-card'><span style='font-size:1.6rem;font-weight:650;'>{max(co2_avoided,0):,.2f} kg</span>"
            f"<div style='font-size:0.85rem;opacity:0.8;'>Emissions avoided by using on-site solar instead of grid.</div></div>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # LINE CHART – LAST 24H, LAST HOUR, FORECAST
    # -----------------------------------------------------
    st.markdown("#### Trend view – recent & forecasted load")

    chart_df = pd.DataFrame(
        {
            "Point": ["24 hours ago", "1 hour ago", "Next hour (forecast)"],
            "Energy (kWh)": [lag24, lag1, pred_kwh],
        }
    )

    # Use Altair (bundled with Streamlit) for a clean line chart
    import altair as alt

    line_chart = (
        alt.Chart(chart_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Point", sort=None, title="Time reference"),
            y=alt.Y("Energy (kWh)", title="Energy"),
            tooltip=["Point", alt.Tooltip("Energy (kWh)", format=".2f")],
        )
        .properties(height=260)
    )

    st.altair_chart(line_chart, use_container_width=True)

    # -----------------------------------------------------
    # NARRATIVE + AI RECOMMENDATIONS
    # -----------------------------------------------------
    st.markdown("### What this scenario shows")

    bullet_lines = [
        f"For **{target_dt.strftime('%Y-%m-%d')} at {hour:02d}:00**, "
        f"PeakGuard AI forecasts **~{pred_kwh:,.1f} kWh**.",
    ]

    if contract_peak_limit_kw > 0:
        if "BREACH" in peak_status:
            bullet_lines.append(
                f"With your contract limit of **{contract_peak_limit_kw:,.1f} kW**, "
                f"status is **🚨 BREACH** – forecast exceeds contract peak by "
                f"~**{delta_kw:,.1f} kW**; high demand charges are likely."
            )
        else:
            bullet_lines.append(
                f"With your contract limit of **{contract_peak_limit_kw:,.1f} kW**, "
                f"status is **✅ SAFE**, with about **{abs(delta_kw):,.1f} kW** headroom."
            )

    if onsite_solar_kw > 0:
        bullet_lines.append(
            f"With **{onsite_solar_kw:,.0f} kW** of solar, the model estimates ~**{solar_kwh:,.1f} kWh** "
            f"on-site generation this hour."
        )
        bullet_lines.append(
            f"That cuts grid draw from **{grid_without_solar_kwh:,.1f} → {grid_with_solar_kwh:,.1f} kWh**, "
            f"avoiding about **{max(co2_avoided,0):,.1f} kg CO₂** in just one hour."
        )

    st.markdown(
        "<ul>" + "".join([f"<li>{line}</li>" for line in bullet_lines]) + "</ul>",
        unsafe_allow_html=True,
    )

    # AI Recommendations box
    recs = generate_ai_recommendations(
        forecast_kwh=pred_kwh,
        contract_limit_kw=contract_peak_limit_kw,
        solar_kw=onsite_solar_kw,
        grid_emission_factor=grid_emission_factor,
        grid_without_solar=grid_without_solar_kwh,
        grid_with_solar=grid_with_solar_kwh,
        co2_avoided_kg=co2_avoided,
    )

    st.markdown("### 🤖 AI recommendations for this hour")

    st.markdown(
        "<div class='sub-card ai-rec'><ul>"
        + "".join([f"<li>{r}</li>" for r in recs])
        + "</ul></div>",
        unsafe_allow_html=True,
    )


else:
    st.markdown(
        "<div class='section-subtitle' style='margin-top:0.6rem;'>"
        "Set your conditions on the left and above, then click "
        "<strong>Run forecast & scenario analysis</strong> to see peaks, solar impact and CO₂ savings."
        "</div>",
        unsafe_allow_html=True,
    )



