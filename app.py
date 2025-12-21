import joblib
import pandas as pd
import mlflow
import matplotlib.pyplot as plt
from sklearn import pipeline
import streamlit as st
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
from time import sleep

set_config(transform_output="pandas")

import dagshub
dagshub.init(repo_name='Uber_Demand_Prediction',repo_owner='MuktiKsinha',mlflow = True)

# set the dagshub tracking server
mlflow.set_tracking_uri('https://dagshub.com/MuktiKsinha/Uber_Demand_Prediction.mlflow')

# get model name
registered_model_name = 'uber_demand_prediction_model'
stage = "Production"
model_path = f"models:/{registered_model_name}/{stage}"

# load the latest model from model registry
model = mlflow.sklearn.load_model(model_path)

# set the root path
root_path = Path(__file__).parent
# path of the data
plot_data_path = root_path / "data/external/plot_data.csv"
data_path = root_path / "data/processed/test.csv"

# model paths
kmeans_path = root_path / "models/mb_kmeans.joblib"
scaler_path = root_path / "models/scaler.joblib"
encoder_path = root_path / "models/encoder.joblib"
model_path = root_path / "models/model.joblib"

# load the objects
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)
model = joblib.load(model_path)
kmeans = joblib.load(kmeans_path)

# dataset to plot
df_plot = pd.read_csv(plot_data_path)
df = pd.read_csv(data_path, parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")

# Configure page
st.set_page_config(
    page_title="Uber Demand Prediction - NYC",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stTitle { color: #1f77d4; text-align: center; font-size: 2.5rem; font-weight: 600; margin-bottom: 1rem; }
    .info-box { background-color: #f0f2f6; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #1f77d4; }
    .info-box h3 { color: #1f77d4; margin-top: 0; margin-bottom: 0.5rem; }
    .metric-card { background-color: #ffffff; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 0.5rem; }
    label { color: #1f1f1f !important; font-weight: 600 !important; font-size: 0.95rem !important; }
    .stDateInput > div > div > input, .stTimeInput > div > div > input { background-color: #ffffff !important; color: #1f1f1f !important; }
    </style>
""", unsafe_allow_html=True)

# UI of app
st.markdown("<h1 style='text-align: center; color: #1f77d4;'>🚕 Uber Demand Prediction - NYC</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Real-time demand forecasting across New York City regions</p>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Target ONLY the radio label text */
    section[data-testid="stSidebar"] div > label {
        color: white !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar configuration
st.sidebar.markdown("### ⚙️ Configuration")
map_type = st.sidebar.radio(
    label="Select Map Type",
    options=["Complete NYC Map", "Only for Neighborhood Regions"],
    index=1,
    help="Choose between viewing all regions or only nearby neighborhoods"
)
st.sidebar.markdown("---")

# Input section with columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='info-box info-box-date'>
        <h3>📅 Date Selection</h3>
        <p style='color: #1f1f1f; font-weight: 600; font-size: 1rem; margin: 0;'>Select the date for demand prediction</p>
    </div>
    """, unsafe_allow_html=True)
    date = st.date_input("Select the date", value=None, min_value=dt.date(2016,3,1), max_value=dt.date(2016,3,31), label_visibility="collapsed")

with col2:
    st.markdown("""
    <div class='info-box info-box-time'>
        <h3>🕐 Time Selection</h3>
        <p style='color: #1f1f1f; font-weight: 600; font-size: 1rem; margin: 0;'>Select the time for demand prediction</p>
    </div>
    """, unsafe_allow_html=True)
    time = st.time_input("Select the time", value=None, label_visibility="collapsed")

# Display selected inputs
if date and time:
    st.markdown("---")
    index = pd.Timestamp(f"{date} {time}")

    # Display selected info in columns
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #666; margin: 0; font-size: 0.9rem;'>Selected Date & Time</p>
            <p style='font-size: 1.3rem; font-weight: 700; color: #1f77d4; margin: 0.8rem 0;'>{index.strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

    delta = dt.timedelta(minutes=15)
    next_interval = index + delta
    with info_col2:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #666; margin: 0; font-size: 0.9rem;'>Forecast Time Interval</p>
            <p style='font-size: 1.3rem; font-weight: 700; color: #1f77d4; margin: 0.8rem 0;'>{next_interval.strftime('%H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

    sample_loc = df_plot.sample(1).reset_index(drop=True)
    lat = sample_loc["pickup_latitude"].item()
    long = sample_loc["pickup_longitude"].item()
    region = sample_loc["region"].item()
    with info_col3:
        st.markdown(f"""
        <div class='metric-card'>
            <p style='color: #666; margin: 0; font-size: 0.9rem;'>Your Region ID</p>
            <p style='font-size: 1.3rem; font-weight: 700; color: #1f77d4; margin: 0.8rem 0;'>{int(region)}</p>
        </div>
        """, unsafe_allow_html=True)

    # Location details in expandable section
    with st.expander("📍 Current Location Details", expanded=False):
        loc_col1, loc_col2 = st.columns(2)
        with loc_col1: 
            st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #666; margin: 0; font-size: 0.9rem;'>Latitude</p>
                <p style='font-size: 1.2rem; font-weight: 700; color: #1f77d4; margin: 0.5rem 0;'>{lat:.6f}</p>
            </div>
            """, unsafe_allow_html=True)
        with loc_col2: 
            st.markdown(f"""
            <div class='metric-card'>
                <p style='color: #666; margin: 0; font-size: 0.9rem;'>Longitude</p>
                <p style='font-size: 1.2rem; font-weight: 700; color: #1f77d4; margin: 0.5rem 0;'>{long:.6f}</p>
            </div>
            """, unsafe_allow_html=True)

    with st.spinner("🔍 Fetching region information..."): sleep(2)
    scaled_cord = scaler.transform(sample_loc.iloc[:, 0:2])

    # plot the map
    st.markdown("---")
    st.markdown("<h2 style='color: #1f77d4;'>🗺️ Demand Map</h2>", unsafe_allow_html=True)

    # colors
    colors = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F", 
              "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00", 
              "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF", 
              "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3", 
              "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347", 
              "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]

    region_colors = {r: colors[i] for i,r in enumerate(df_plot["region"].unique().tolist())}
    df_plot["color"] = df_plot["region"].map(region_colors)

    pipe = Pipeline([('encoder', encoder), ('reg', model)])

    if map_type == "Complete NYC Map":
        # progress bar
        progress_bar = st.progress(0, text="Loading map data...")
        for p in range(100):
            sleep(0.05)
            progress_bar.progress(p+1, text="Loading map data...")
        st.map(df_plot, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        input_data = df.loc[index, :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Demand by Region")

        # Find max demand
        max_idx = predictions.argmax()

        # Create legend
        legend_cols = st.columns(3)
        for ind in range(len(predictions)):
            col_idx = ind % 3
            with legend_cols[col_idx]:
                color = colors[ind]
                demand = predictions[ind]
                current_indicator = "📍 Current" if region == ind else ""
                max_indicator = "🔥 Highest Demand" if ind == max_idx else ""
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: {'1rem' if ind==max_idx else '0.5rem'};
                    border-radius: 5px;
                    margin: 0.3rem 0;
                    color: white;
                    font-weight: 600;
                    box-shadow: {'0 0 15px 5px #FFD700' if ind==max_idx else 'none'};
                    border: {'2px solid #FFD700' if ind==max_idx else 'none'};
                ">
                    Region {int(ind)} {current_indicator} {max_indicator}<br>
                    <span style="font-size: 0.9rem;">Demand: {int(demand)} pickups</span>
                </div>
                """, unsafe_allow_html=True)

    elif map_type == "Only for Neighborhood Regions":
        distances = list(enumerate(kmeans.transform(scaled_cord).values.ravel()))
        sorted_distances = sorted(distances, key=lambda x: x[1])[0:9]
        indexes = sorted([ind[0] for ind in sorted_distances])

        df_plot_filtered = df_plot[df_plot["region"].isin(indexes)]

        progress_bar = st.progress(0, text="Loading neighborhood map...")
        for p in range(100):
            sleep(0.05)
            progress_bar.progress(p+1, text="Loading neighborhood map...")
        st.map(df_plot_filtered, latitude="pickup_latitude", longitude="pickup_longitude", size=0.01, color="color")
        progress_bar.empty()

        input_data = df.loc[index, :]
        input_data = input_data.loc[input_data["region"].isin(indexes), :].sort_values("region")
        predictions = pipe.predict(input_data.drop(columns=["total_pickups"]))

        st.markdown("### Demand in Nearby Neighborhoods")

        max_idx = predictions.argmax()

        legend_cols = st.columns(3)
        for ind in range(len(predictions)):
            col_idx = ind % 3
            with legend_cols[col_idx]:
                color = colors[indexes[ind]]
                demand = predictions[ind]
                current_indicator = "📍 Current" if region == indexes[ind] else ""
                max_indicator = "🔥 Highest Demand" if ind == max_idx else ""
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: {'1rem' if ind==max_idx else '0.5rem'};
                    border-radius: 5px;
                    margin: 0.3rem 0;
                    color: white;
                    font-weight: 600;
                    box-shadow: {'0 0 15px 5px #FFD700' if ind==max_idx else 'none'};
                    border: {'2px solid #FFD700' if ind==max_idx else 'none'};
                ">
                    Region {int(indexes[ind])} {current_indicator} {max_indicator}<br>
                    <span style="font-size: 0.9rem;">Demand: {int(demand)} pickups</span>
                </div>
                """, unsafe_allow_html=True)



