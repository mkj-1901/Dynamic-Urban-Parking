import numpy as np
import pandas as pd
import streamlit as st
import requests
from datetime import datetime
from streamlit_bokeh import streamlit_bokeh
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from pydantic_models import meta_dict, vehicle_map

# === Page Configuration ===
st.set_page_config(page_title="Smart Parking Price", layout="wide")

# === Custom CSS ===
st.markdown("""
    <style>
        .main-title {
            font-size: 1.8em;
            font-weight: bold;
            color: #888;
            text-align: center;
            margin-bottom: 10px;
        }
        .form-container {
            max-width: 300px;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# === Header ===
st.markdown("<div class='main-title'>🚘 Dynamic Urban Parking Price Predictor</div>", unsafe_allow_html=True)

# === Navigation State ===
if 'selected' not in st.session_state:
    st.session_state.selected = "predict"

nav = st.columns([1, 1], gap="small")
with nav[0]:
    if st.button("🚗 Predict", use_container_width=True):
        st.session_state.selected = "predict"
with nav[1]:
    if st.button("📊 Visualize", use_container_width=True):
        st.session_state.selected = "visualize"

selected = st.session_state.selected

# === Predict Section ===
if selected == "predict":
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.subheader("🚗 Dynamic Parking Price Predictor")

    scn = st.selectbox(
        "Select Parking Slot",
        options=meta_dict.keys(),
        format_func=lambda k: meta_dict[k]["SystemCodeNumber"]
    )

    vtype = st.selectbox(
        "Select Vehicle Type",
        options=vehicle_map.keys(),
        format_func=lambda k: vehicle_map[k]
    )

    date = st.date_input("Select Date", value=datetime(2025, 7, 7).date())
    time = st.time_input("Select Time", value=datetime(2025, 7, 7, 10, 0).time())
    timestamp = datetime.combine(date, time).isoformat()

    if st.button("🔮 Predict Price"):
        try:
            response = requests.post(
                url="https://dynamic-urban-parking-backend.onrender.com/predict",
                json={
                    "SystemCodeNumber": scn,
                    "TimeStamp": timestamp,
                    "VehicleType": vtype
                }
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"💰 Predicted Parking Price: ₹{result['price']}")
                st.info(f"🚣️ Expected Traffic: {result['ExpectedTrafficNearby']}")
                if result["reroute_to"]:
                    st.warning(f"📍 Suggested Alternate Slot: {result['reroute_to']}")
                else:
                    st.info("✅ No rerouting needed.")
            else:
                st.error(f"❌ API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

elif selected == "visualize":
    st.subheader("📊 Parking Data Visualizations")
    st.subheader("🗺️ Interactive Bokeh Visualization")

    st.markdown("### Dynamic Parking Price Plot")


    @st.cache_data
    def load_stream_data():
        try:
            df = pd.read_csv("output/output_price_stream.csv")
            df['t'] = pd.to_datetime(df['t'], errors='coerce')
            df = df.dropna(subset=['t'])
            df = df[df['system_id'].between(0, 13)]
            df['system_id'] = df['system_id'].astype(int)
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()


    ops = load_stream_data()

    if ops.empty:
        st.warning("No data available for visualization.")
    else:
        unique_ids = sorted(ops['system_id'].unique())

        if not unique_ids:
            st.warning("No valid system IDs found in data.")
        else:
            selected_ids = st.multiselect(
                "Select Parking SystemCodeNumbers",
                options=unique_ids,
                default=[unique_ids[0]],
                key="bokeh_multiselect"
            )

            if not selected_ids:
                st.warning("Please select at least one SystemCodeNumber")
            else:
                plot = figure(
                    title="Parking Price Over Time",
                    x_axis_type='datetime',
                    width=800,
                    height=400,
                    tools="pan,wheel_zoom,box_zoom,reset,save",
                )

                for sid in selected_ids:
                    df_sid = ops[ops['system_id'] == sid].sort_values('t')
                    if not df_sid.empty:
                        source = ColumnDataSource(df_sid)
                        plot.line(
                            x='t',
                            y='price',
                            source=source,
                            legend_label=f"System {sid}",
                            line_width=2
                        )

                plot.legend.location = "top_left"
                plot.xaxis.axis_label = "Time"
                plot.yaxis.axis_label = "Price (₹)"

                # ✅ Render the plot using streamlit_bokeh
                streamlit_bokeh(plot)

    st.header("📈 Exploratory Data Analysis (EDA)")
    st.image("output/Latitude_Longitude.png", caption="Distribution of Parking Spots", use_container_width=True)
    st.image("output/QueueLength_AvgTrafficConditionNearby.png", caption="QueueLength vs AvgTrafficConditionNearby",
             use_container_width=True)
    st.image("output/QueueLength_SpecialDay.png", caption="QueueLength vs SpecialDay", use_container_width=True)
    st.image("output/QueueLength_TimeofDay.png", caption="QueueLength vs Time of Day", use_container_width=True)

    st.header("🤖 Model Comparison")
    st.image("output/Model1.png", caption="Model 1 - Price with time", use_container_width=False)
    st.image("output/Model2.png", caption="Model 2 - Price with time", use_container_width=False)
    st.image("output/Model3.png", caption="Model 3 - Price with time", use_container_width=False)