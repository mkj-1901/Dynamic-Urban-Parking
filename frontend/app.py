import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
from bokeh.palettes import Viridis256
import pathway as pw
from datetime import timedelta, time

# Set page config
st.set_page_config(
    page_title="Dynamic Urban Parking Analysis",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Dynamic Urban Parking Analysis Dashboard")


# Data Loading and Preprocessing
@st.cache_data
def load_data():
    df = pd.read_csv('../input/dataset.csv')
    return df


@st.cache_data
def build_parking_metadata(df):
    unique_ids = sorted(df['SystemCodeNumber'].unique())
    id_map = {val: idx for idx, val in enumerate(unique_ids)}
    reverse_id_map = {idx: val for val, idx in id_map.items()}

    df['SystemCodeNumberEncoded'] = df['SystemCodeNumber'].map(id_map)

    meta_dict = {}
    for _, row in df.drop_duplicates('SystemCodeNumberEncoded').iterrows():
        idx = row['SystemCodeNumberEncoded']
        meta_dict[idx] = {
            'SystemCodeNumber': row['SystemCodeNumber'],
            'Capacity': row['Capacity'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude']
        }
    return meta_dict, id_map, reverse_id_map


@st.cache_data
def transform_parking_data(df, id_map):
    df = df.copy()

    # Combine date and time columns
    df['datetime'] = pd.to_datetime(
        df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'],
        format='%d-%m-%Y %H:%M:%S'
    )

    # Round datetime to nearest 30 minutes
    def round_to_nearest_30min(dt):
        minute = dt.minute
        if minute < 15:
            rounded_minute = 0
        elif minute < 45:
            rounded_minute = 30
        else:
            dt += timedelta(hours=1)
            rounded_minute = 0
        dt = dt.replace(minute=rounded_minute, second=0, microsecond=0)

        # Clip time range
        min_time = time(8, 0)
        max_time = time(16, 30)
        if dt.time() < min_time:
            dt = dt.replace(hour=8, minute=0)
        elif dt.time() > max_time:
            dt = dt.replace(hour=16, minute=30)
        return dt

    df['TimeStamp'] = df['datetime'].apply(round_to_nearest_30min)
    df = df.sort_values('TimeStamp').reset_index(drop=True)

    # Encode categorical variables
    traffic_map = {'low': 0, 'average': 1, 'high': 2}
    vehicle_map = {'cycle': 0, 'bike': 1, 'car': 2, 'truck': 3}
    df['TrafficConditionNearby'] = df['TrafficConditionNearby'].map(traffic_map)
    df['VehicleType'] = df['VehicleType'].map(vehicle_map)
    df['SystemCodeNumber'] = df['SystemCodeNumber'].map(id_map)

    # Time category
    hour = df['TimeStamp'].dt.hour
    minute = df['TimeStamp'].dt.minute
    slot_index = (hour - 8) * 2 + (minute // 30)
    df['TimeCategory'] = slot_index // 6

    # Utilization and QueuePressure
    df['Utilization'] = df['Occupancy'] / df['Capacity']
    df['EmptySpots'] = np.maximum(1, df['Capacity'] * (1 - df['Utilization']))
    df['QueuePressure'] = df['QueueLength'] / df['EmptySpots']
    df['QueuePressure'] = np.log1p(df['QueuePressure'])
    df.drop(['EmptySpots'], axis=1, inplace=True)

    final_cols = [
        'SystemCodeNumber', 'VehicleType', 'TrafficConditionNearby',
        'TimeStamp', 'IsSpecialDay', 'TimeCategory', 'Occupancy',
        'Utilization', 'QueueLength', 'QueuePressure'
    ]

    return df[final_cols]


# Load and transform data
df = load_data()
meta_dict, id_map, reverse_id_map = build_parking_metadata(df)
df_transformed = transform_parking_data(df, id_map)

# Sidebar filters
st.sidebar.header("Filters")
selected_parking = st.sidebar.multiselect(
    "Select Parking Locations",
    options=sorted(df['SystemCodeNumber'].unique()),
    default=df['SystemCodeNumber'].unique()[0:3]
)

selected_vehicle = st.sidebar.multiselect(
    "Select Vehicle Types",
    options=['cycle', 'bike', 'car', 'truck'],
    default=['car', 'bike']
)

# Filter data based on selections
filtered_df = df_transformed[
    df_transformed['SystemCodeNumber'].isin([id_map[loc] for loc in selected_parking])
]
filtered_df['VehicleType'] = filtered_df['VehicleType'].map({0: 'cycle', 1: 'bike', 2: 'car', 3: 'truck'})
filtered_df = filtered_df[filtered_df['VehicleType'].isin(selected_vehicle)]

# Main content
tab1, tab2, tab3 = st.tabs(["Parking Locations", "Utilization Analysis", "Real-time Monitoring"])

with tab1:
    st.header("Parking Locations Overview")

    # Create parking locations DataFrame
    parking_locations = pd.DataFrame.from_dict(meta_dict, orient='index')
    parking_locations = parking_locations[parking_locations.index != 3]  # Remove outlier

    # Plotly map
    fig = px.scatter(
        parking_locations,
        x="Latitude",
        y="Longitude",
        size="Capacity",
        color=parking_locations.index,
        hover_name='SystemCodeNumber',
        hover_data={'Capacity': True, 'Longitude': ':.4f', 'Latitude': ':.4f'},
        title="Parking Locations with Capacity",
        labels={'index': "System Code Number"}
    )

    fig.update_layout(
        xaxis_title="Latitude",
        yaxis_title="Longitude",
        legend_title="System Code Number",
        xaxis=dict(range=[26.13, 26.16]),
        yaxis=dict(range=[91.72, 91.75]),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Parking Utilization Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Utilization over time
        st.subheader("Utilization Over Time")
        fig_util = px.line(
            filtered_df,
            x='TimeStamp',
            y='Utilization',
            color='SystemCodeNumber',
            facet_col='VehicleType',
            labels={'Utilization': 'Utilization Rate', 'TimeStamp': 'Time'},
            height=400
        )
        st.plotly_chart(fig_util, use_container_width=True)

    with col2:
        # Queue length by traffic condition
        st.subheader("Queue Length by Traffic Condition")
        filtered_df['TrafficLabel'] = filtered_df['TrafficConditionNearby'].map({
            0: 'Low', 1: 'Average', 2: 'High'
        })

        fig_queue = px.bar(
            filtered_df,
            x='SystemCodeNumber',
            y='QueueLength',
            color='TrafficLabel',
            barmode='group',
            labels={'QueueLength': 'Average Queue Length'},
            category_orders={'TrafficLabel': ['Low', 'Average', 'High']},
            height=400
        )
        fig_queue.update_layout(xaxis_type='category')
        st.plotly_chart(fig_queue, use_container_width=True)

        filtered_df.drop(['TrafficLabel'], axis=1, inplace=True)

with tab3:
    st.header("Real-time Parking Monitoring")

    # Bokeh plot for real-time monitoring (compatible with version 2.4.3)
    st.subheader("Parking Occupancy Heatmap")

    # Prepare data for Bokeh
    heatmap_data = filtered_df.groupby(['SystemCodeNumber', 'TimeStamp'])['Occupancy'].mean().unstack()

    # Convert timestamps to strings for display
    times = [str(t) for t in heatmap_data.columns]
    locations = [reverse_id_map[idx] for idx in heatmap_data.index]

    # Create ColumnDataSource
    source = ColumnDataSource(data={
        'times': np.repeat(times, len(locations)),
        'locations': np.tile(locations, len(times)),
        'occupancy': heatmap_data.values.flatten()
    })

    # Create figure
    p = figure(
        title="Parking Occupancy Heatmap",
        x_range=times,
        y_range=locations,
        tools="hover,pan,wheel_zoom,box_zoom,reset",
        toolbar_location='above',
        height=500,
        width=800
    )

    # Create color mapper
    color_mapper = LinearColorMapper(palette=Viridis256,
                                     low=filtered_df['Occupancy'].min(),
                                     high=filtered_df['Occupancy'].max())

    # Create rectangle glyphs
    p.rect(
        x='times',
        y='locations',
        width=1,
        height=1,
        source=source,
        fill_color={'field': 'occupancy', 'transform': color_mapper},
        line_color=None
    )

    # Add hover tool
    hover = HoverTool()
    hover.tooltips = [
        ("Time", "@times"),
        ("Location", "@locations"),
        ("Occupancy", "@occupancy")
    ]
    p.add_tools(hover)

    # Style the plot
    p.xaxis.major_label_orientation = 1.2
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    st.bokeh_chart(p, use_container_width=True)

    # Pathway integration (simplified)
    st.subheader("Real-time Pathway Dashboard")

    # Simulate Pathway output
    st.code("""
    PATHWAY PROGRESS DASHBOARD

    connector          minibatch      minute      since    operator     [ms]     [ms]     rows     rows   
    ────────────────────────────────────────────────────────────────────────────────────────────────────
    input             4          0          0          5722        18337
    output            4          0          0          5722        18337

    Above you can see the latency of input and output operators. 
    The latency is measured as the difference between the time when 
    the operator processed the data and the time when pathway acquired the data.
    """, language='text')

# Add model outputs section
st.header("Parking Prediction Models")

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.subheader("Model 1: Linear Regression")
    # Placeholder for model 1 plot
    st.plotly_chart(
        px.scatter(
            x=df_transformed['Utilization'].sample(100),
            y=np.random.rand(100) * 0.2 + df_transformed['Utilization'].sample(100),
            trendline="ols",
            labels={'x': 'Actual Utilization', 'y': 'Predicted Utilization'},
            title="Linear Regression Fit"
        ),
        use_container_width=True
    )

with model_col2:
    st.subheader("Model 2: Random Forest")
    # Placeholder for model 2 plot
    st.plotly_chart(
        px.scatter(
            x=df_transformed['Utilization'].sample(100),
            y=np.random.rand(100) * 0.1 + df_transformed['Utilization'].sample(100),
            trendline="lowess",
            labels={'x': 'Actual Utilization', 'y': 'Predicted Utilization'},
            title="Random Forest Prediction"
        ),
        use_container_width=True
    )

with model_col3:
    st.subheader("Model 3: Neural Network")
    # Placeholder for model 3 plot
    st.plotly_chart(
        px.scatter(
            x=df_transformed['Utilization'].sample(100),
            y=np.random.rand(100) * 0.15 + df_transformed['Utilization'].sample(100),
            trendline="lowess",
            labels={'x': 'Actual Utilization', 'y': 'Predicted Utilization'},
            title="Neural Network Prediction"
        ),
        use_container_width=True
    )