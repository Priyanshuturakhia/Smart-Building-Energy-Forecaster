import streamlit as st
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib

# --- 1. SET PAGE CONFIG (Sets browser tab title and icon) ---
st.set_page_config(page_title="PeakGuard AI", page_icon="💡")


# --- 2. LOAD YOUR MODEL AND FILES (at startup) ---
# Use @st.cache_resource to load these only once
@st.cache_resource
def load_model_and_files():
    try:
        model = lgb.Booster(model_file='lgbm_model.txt')
        model_features = joblib.load('feature_names.pkl')
        all_primary_uses = [col.replace('primary_use_', '') for col in model_features if 'primary_use_' in col]
    except FileNotFoundError:
        st.error(
            "Error: Model or feature files not found. Make sure 'lgbm_model.txt' and 'feature_names.pkl' are in the same folder.")
        st.stop()
    return model, model_features, all_primary_uses


model, model_features, all_primary_uses = load_model_and_files()


# --- 3. DEFINE THE FEATURE ENGINEERING FUNCTION (Same as before) ---
def create_features_from_input(air_temp, hour, day_of_week, month, primary_use, square_feet, year_built, lag1, lag24):
    input_data = {
        'air_temperature': air_temp, 'square_feet': square_feet, 'year_built': year_built,
        'month': month, 'meter_reading_lag1': lag1, 'meter_reading_lag24': lag24
    }
    input_data['hdd'] = max(0, 18 - air_temp)  # 18 was comfort_base_heating
    input_data['cdd'] = max(0, air_temp - 21)  # 21 was comfort_base_cooling
    input_data['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    input_data['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    input_data['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7.0)
    input_data['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7.0)

    for use in all_primary_uses:
        input_data[f'primary_use_{use}'] = 1 if use == primary_use else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_features, fill_value=0)
    return input_df


# --- 4. STREAMLIT UI ---

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Building Information")
    primary_use = st.selectbox("Primary Building Use", all_primary_uses, key="primary_use")
    square_feet = st.number_input("Building Square Feet", 10000, 500000, 50000, key="sqft")
    year_built = st.number_input("Year Built", 1950, 2024, 2005, key="year")
    st.info(
        "💡 **Note:** This model is designed for commercial properties and may not be accurate for single-family homes.")

# --- Main Page Title ---
st.title("💡 PeakGuard AI")
st.write("Enter the building's current conditions to forecast its energy consumption for the next hour.")

# --- Prediction Output Area ---
# We create a placeholder for the metric. This is good practice.
prediction_output = st.empty()

# --- Main Page Inputs (Inside a Form) ---
with st.form(key="prediction_form"):
    st.header("Weather & Time Inputs")

    # Use columns for a cleaner layout
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.slider("Air Temperature (°C)", -10.0, 40.0, 25.0, key="temp")
        day_of_week = st.slider("Day of the Week (0=Mon, 6=Sun)", 0, 6, 0, key="day")
    with col2:
        hour = st.slider("Hour of the Day", 0, 23, 14, key="hour")
        month = st.slider("Month", 1, 12, 7, key="month")

    # Use an expander for "advanced" features
    with st.expander("Advanced: Lag Features (Optional)"):
        st.warning(
            "**Note:** In a real app, this data would be fetched automatically from a database. These are the *most important* features for an accurate prediction.",
            icon="⚠️")
        lag1 = st.number_input("Energy 1 Hour Ago (kWh)", 0.0, 10000.0, 100.0, step=10.0, key="lag1")
        lag24 = st.number_input("Energy 24 Hours Ago (kWh)", 0.0, 10000.0, 110.0, step=10.0, key="lag24")

    # The single submit button for the form
    submit_button = st.form_submit_button(label="Predict Consumption")

# --- 5. RUN PREDICTION (Only when form is submitted) ---
if submit_button:
    # 1. Create all features from the simple inputs
    input_df = create_features_from_input(
        air_temp, hour, day_of_week, month, primary_use,
        square_feet, year_built, lag1, lag24
    )

    # 2. Make prediction
    prediction_log = model.predict(input_df)

    # 3. Reverse the log-transform
    prediction = np.expm1(prediction_log[0])

    # 4. Display the prediction in the 'empty' placeholder
    prediction_output.metric(
        label="Predicted Energy Consumption",
        value=f"{prediction:.2f} kWh"

    )
