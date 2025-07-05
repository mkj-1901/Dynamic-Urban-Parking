import streamlit as st
import requests
from datetime import datetime, date, time, timedelta

# ------------------------
# Configuration
# ------------------------
API_URL = "http://127.0.0.1:8000/predict"

id_map = {
    'BHMBCCMKT01': 0, 'BHMBCCTHL01': 1, 'BHMEURBRD01': 2, 'BHMMBMMBX01': 3,
    'BHMNCPHST01': 4, 'BHMNCPNST01': 5, 'Broad Street': 6, 'Others-CCCPS105a': 7,
    'Others-CCCPS119a': 8, 'Others-CCCPS135a': 9, 'Others-CCCPS202': 10,
    'Others-CCCPS8': 11, 'Others-CCCPS98': 12, 'Shopping': 13
}

vehicle_label_to_code = {'Cycle': 0, 'Bike': 1, 'Car': 2, 'Truck': 3}
vehicle_code_to_label = {v: k for k, v in vehicle_label_to_code.items()}

# ------------------------
# Helpers
# ------------------------
def clip_and_round_time(ts: datetime) -> tuple[datetime, str]:
    original_ts = ts
    min_t = time(8, 0)
    max_t = time(16, 30)

    # Clip to range
    clipped = False
    if ts.time() < min_t:
        ts = ts.replace(hour=8, minute=0)
        clipped = True
    elif ts.time() > max_t:
        ts = ts.replace(hour=16, minute=30)
        clipped = True

    # Round to nearest 30 minutes
    minute = ts.minute
    if minute < 15:
        rounded_minute = 0
    elif minute < 45:
        rounded_minute = 30
    else:
        ts += timedelta(hours=1)
        rounded_minute = 0
    ts = ts.replace(minute=rounded_minute, second=0, microsecond=0)

    rounded = ts != original_ts
    note = ""
    if clipped:
        note += "⏱️ Time was clipped to 08:00–16:30.\n"
    if rounded:
        note += f"🕓 Rounded to nearest 30 min: **{ts.strftime('%H:%M')}**"
    return ts, note.strip()

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="Smart Parking Predictor", layout="centered")
st.title("🚗 Smart Parking Price Predictor")

with st.form("prediction_form"):
    spot_name = st.selectbox("📍 Select Parking Spot", options=list(id_map.keys()))
    vehicle_label = st.selectbox("🚘 Vehicle Type", options=list(vehicle_label_to_code.keys()))
    selected_date = st.date_input("📅 Date", value=date(2023, 7, 5), format="MM/DD/YYYY")
    selected_time = st.time_input("⏰ Time", value=time(12, 0))
    submit_button = st.form_submit_button(label="🔮 Predict")

# ------------------------
# Prediction Logic
# ------------------------
if submit_button:
    try:
        vehicle_type = vehicle_label_to_code[vehicle_label]
        combined_ts = datetime.combine(selected_date, selected_time).replace(year=2023)
        final_ts, msg = clip_and_round_time(combined_ts)

        if msg:
            st.warning(msg)

        payload = {
            "SystemCodeNumber": id_map[spot_name],
            "VehicleType": vehicle_type,
            "TimeStamp": final_ts.isoformat()
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()
            st.success("✅ Prediction Successful!")

            st.write("### 🔎 Results")
            st.metric("📊 Traffic Nearby", result["TrafficConditionNearby"])  # Already a string
            st.metric("🚦 Queue Length", result["QueueLength"])
            st.metric("📈 Utilization", f"{result['Utilization']:.2f}")
            st.metric("💸 Suggested Price", f"₹{result['price']}")
            st.metric("🔁 Reroute To", result["RerouteTo"] if result["RerouteTo"] is not None else "None")

        else:
            try:
                st.error(f"❌ Server Error: {response.json().get('error', 'No details')}")
            except Exception:
                st.error("❌ Server Error with no JSON response")

    except Exception as e:
        st.error(f"⚠️ Exception occurred: {e}")
