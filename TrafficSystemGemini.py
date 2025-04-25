# Directory structure:
# traffic_app/
# ├── Home.py            <- Home dashboard (general info, KPIs)
# ├── Prediction.py      <- Predict traffic density
# ├── Visualizations.py  <- User-chosen visualizations
# ├── Reports.py         <- Export CSV/PDF with selected filters
# ├── utils.py           <- Shared functions like model loading, mappings, etc.
# ├── catmodel_traffic_model.pkl
# ├── data/
# └──   └── cleaned_urban_traffic_density.csv
# └── img/   
# └──   └── city1_xuCMblCiC.gif
#       ├── 1709808629051.png
#       ├── 9583344.gif
#       ├── 9909421_1577692269_banner of smart city.jpg
#       ├── city1_xuCMblCiC.gif
#       └── ©-shutterstockmetamorworks_2082107314.jpg
# └── traffic_preprocessor.pkl
# └── requirements.txt




import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import datetime
import matplotlib.pyplot as plt
from fpdf import FPDF
from io import BytesIO
import os
import requests
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import openai
import re
import threading
import time
import random
import folium
from streamlit_folium import folium_static
import gdown
import google.generativeai as genai
from utils import generate_explanation



# -----------------------------
# SETUP & CONFIGURATION
# -----------------------------

#GEMINI_API_KEY = "AIzaSyAF_dpw5T3INCM-y9NNZW8z9uaEbdizwps"
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# Streamlit Page Config
st.set_page_config(page_title="🚦 Smart City Traffic System", layout="wide", initial_sidebar_state="expanded")



# Define Relative Paths for Deployment
MODEL_PATHS = {
    "CatBoost": "C:/Users/Ning Sheng Yong/Desktop/QING APU/catmodel_traffic_model.pkl"
}
#DATA_PATH = "https://drive.google.com/uc?export=download&id=1cJcWoYNuhKNWluzd4mBsScuw0lHIhs5g"
DATA_PATH = "C:/Users/Ning Sheng Yong/Desktop/QING APU/cleaned_urban_traffic_density.csv"
PREPROCESSOR_PATH = "C:/Users/Ning Sheng Yong/Desktop/QING APU/traffic_preprocessor.pkl"


# -----------------------------
# SIDEBAR & NAVIGATION
# -----------------------------
st.sidebar.title("Navigation")
role = st.sidebar.selectbox("User Role", ["General User", "City Planner", "Admin"])
page = st.sidebar.radio("Go to", ["🏠 Home", "📈 Prediction", "📊 Visualizations", "📥 Reports", "💬 Feedback", "ℹ️ About"])

with st.sidebar.expander("About this App"):
    st.markdown("""
    **Smart City Traffic System** provides insights into urban traffic conditions.
    
    **Key Features:**
    - **Predictions:** Tntegrated departure time advice.
    - **Comparative Analytics & Reports:** Detailed visualizations and PDF reports.
    - **Feedback System:** Public and City Planners can submit feedback; Admins view and delete feedback.
    
    **Access Levels:**
    - *General User:* View predictions, use simulation controls and submit feedback.
    - *City Planner:* Access public features plus detailed reports/analytics.
    - *Admin:* Manage models/data and view/delete all submitted feedback.
    """)


# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def load_all_models(model_paths):
    models = {}
    for name, path in model_paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


def get_time_of_day(hour):
    if 0 <= hour <= 5:
        return 'Night'
    elif 6 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 16:
        return 'Afternoon'
    elif 17 <= hour <= 20:
        return 'Evening'
    else:
        return 'Night'

def get_weather_category(weather):
    weather = weather.lower()
    if 'clear' in weather:
        return 'Clear'
    if 'rain' in weather:
        return 'Rainy'
    if 'snow' in weather:
        return 'Snowy'
    return 'Other'

def mappings(data):
    return {
        'city': {city: i for i, city in enumerate(sorted(data['City'].unique()))},
        'vehicle': {v: i for i, v in enumerate(sorted(data['Vehicle Type'].unique()))},
        'economic': {e: i for i, e in enumerate(sorted(data['Economic Condition'].unique()))},
        'time_of_day': {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3},
        'weather_category': {'Clear': 0, 'Rainy': 1, 'Snowy': 2, 'Other': 3}
    }


# Strip HTML for PDF text conversion
def strip_html(html):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html)



def safe_pdf_text(txt: str) -> str:
    # replace common dashes with ASCII hyphen
    txt = txt.replace("\u2014", "-").replace("\u2013", "-")
    # drop any remaining non-Latin1 chars
    return txt.encode('latin-1', 'ignore').decode('latin-1')


# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def gemini_generate_explanation(
    prompt: str,
    model_name: str = "gemini-2.0-flash"
) -> str:
    """
    Call Google Gemini to generate a short explanation for a prompt.
    """
    # Directly use client.models.generate_content (no need to get_model)
    response = client.models.generate_content(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": prompt}]}]
    )
        

    # Access the text output safely
    return response.text if hasattr(response, "text") else ""



def generate_explanation(plot_choice: str, filters: dict) -> str:
    """
    Build a concise prompt for the chosen traffic plot and return
    Gemini’s output.
    """
    prompt = (
        f"Explain this traffic visualization ({plot_choice}) for a Smart City Traffic System. "
        f"Filters applied: City={filters['city']}, Vehicle={filters['vehicle']}, "
        f"Weather={filters['weather']}, EconomicCondition={filters['econ']}."
    )
    # strip any accidental HTML tags from Gemini
    explanation = gemini_generate_explanation(prompt)
    return re.sub(r"<.*?>", "", explanation).strip()



# -----------------------------
# MANUAL HOLIDAYS & DATA LOADING
# -----------------------------
manual_holidays = {
    datetime.date(2024, 1, 1),
    datetime.date(2024, 2, 10),
    datetime.date(2024, 5, 1),
    datetime.date(2024, 8, 31),
    datetime.date(2024, 12, 25),
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 29),
}

data = load_data(DATA_PATH)
all_models = load_all_models(MODEL_PATHS)
preprocessor = joblib.load(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else None
maps = mappings(data)



# -----------------------------
# SESSION STATE DEFAULTS
# -----------------------------
for key, default in {
    "city": data['City'].mode()[0],
    "vehicle": data['Vehicle Type'].mode()[0],
    "weather": data['Weather_Category'].mode()[0],
    "econ": data['Economic Condition'].mode()[0],
    "hour": 8,
    "date": datetime.date.today()
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------
# NEW FEATURE: DEPARTURE TIME ADVICE (Dynamic range based on user input)
# -----------------------------
def get_departure_advice(input_df, model, preprocessor, user_hour, waiting_penalty=0.5, horizon=7):
    """
    Calculates candidate departure times from the current hour (user_hour) to (user_hour + horizon)
    and returns the hour with the lowest weighted cost.
    
    The cost function is defined as:
        cost = predicted_congestion + waiting_penalty * (candidate_hour - user_hour)
    
    The function returns:
     - best_hour (wrapped around modulo 24), the predicted congestion for each candidate hour,
     - and the computed cost for each candidate.
    """
    candidate_hours = list(range(user_hour, user_hour + horizon))
    predictions = {}
    costs = {}
    for h in candidate_hours:
        temp = input_df.copy()
        candidate = h % 24  # Ensure hour remains in 0-23.
        temp["Hour Of Day"] = candidate
        temp["Time of Day"] = get_time_of_day(candidate)
        if preprocessor:
            transformed = preprocessor.transform(temp)
        else:
            transformed = temp.values
        # Predict congestion label as an integer (0: 'Very Low', 1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High')
        pred = model.predict(transformed)[0]
        pred_clipped = int(np.clip(pred.item(), 0, 4))
        predictions[h] = pred_clipped
        # Calculate cost: lower congestion is better, but waiting longer adds penalty.
        costs[h] = pred_clipped + waiting_penalty * (h - user_hour)
    best_hour = min(costs, key=costs.get)
    return best_hour % 24, predictions, costs



# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center;'>🚦 Smart City Traffic Dashboard</h1>", unsafe_allow_html=True)

    left, center, right = st.columns([1, 2, 1])  # 1:2:1 ratio for centering
    with center:
        st.image("C:/Users/Ning Sheng Yong/Desktop/QING APU/SystemImages/city1_xuCMblCiC.gif", use_container_width=True)


    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🚗 Avg Speed", f"{data['Speed'].mean():.1f} km/h")
    col2.metric("⚡ Avg Energy", f"{data['Energy Consumption'].mean():.1f} kWh")
    col3.metric("📉 Avg Impact", f"{data['Speed_Traffic_Impact_Label'].mean():.2f}")
    col4.metric("🌦️ Common Weather", data['Weather_Category'].mode()[0])
    
    st.markdown("#### What These Metrics Mean:")
    st.markdown("""
    - **🚗 Avg Speed:** The average vehicle speed, indicating overall traffic flow quality.
                
    - **⚡ Avg Energy:** Average energy consumption per vehicle, reflecting traffic intensity and stop-and-go conditions.
    
    - **📉 Avg Impact:** Mean of predicted traffic congestion levels across the dataset.
   
    - **🌦️ Common Weather:** Most frequent weather condition influencing traffic trends.
    """)

    
    st.markdown("### Welcome to the Smart City Traffic System")
    st.markdown("""
    This dashboard provides an integrated platform to monitor and predict urban traffic conditions.
    
    Use the **Prediction** page to forecast traffic density, explore interactive charts on the **Visualizations** page, 
    and generate detailed reports on the **Reports** page.
    """)
    st.caption("© 2025 Smart City Traffic System | Powered by Streamlit")


# -----------------------------
# PREDICTION PAGE
# -----------------------------
elif page == "📈 Prediction":
    st.markdown("<h1 style='text-align: center;'>📈 Traffic Prediction</h1>", unsafe_allow_html=True)
    
    # Use CatBoost directly as the default and only model
    selected_model = all_models["CatBoost"]

    st.markdown("Using **CatBoost** model for traffic prediction.")
    
    
    # Prediction Inputs
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            city    = st.selectbox("City", sorted(data['City'].unique()), key="city")
            vehicle = st.selectbox("Vehicle Type", sorted(data['Vehicle Type'].unique()), key="vehicle")
            weather = st.selectbox("Weather", sorted(data['Weather_Category'].unique()), key="weather")
        with col2:
            econ = st.selectbox("Economic Condition", 
                                sorted(data['Economic Condition'].unique()), key="econ")
            date = st.date_input("Date", 
                                  value=st.session_state['date'], key="date")
            hour = st.slider("Hour", 0, 23, st.session_state['hour'], key="hour")

            
    random_event = 1 if st.radio("Random Event?", ["No", "Yes"]) == "Yes" else 0

    # ————— Add a little helper text with example events —————
    st.markdown(
    "**If ‘Yes’, a random event could be:** road block | hazard | road closure | accident | vehicle breakdown | stalled car on route"
    )

     # ————— Let the user specify their route distance —————
    distance = st.number_input(
        "Route Distance (km)",
        min_value=0.1,
        value=10.0,
        step=0.1,
        help="Enter the approximate length of your journey."
    )


    
    # Default values based on filters
    subset = data[(data['City'] == city) &
                  (data['Vehicle Type'] == vehicle) &
                  (data['Weather_Category'] == weather) &
                  (data['Economic Condition'] == econ)]
    if not subset.empty:
        default_speed = subset['Speed'].mean()
        default_energy = subset['Energy Consumption'].mean()
    else:
        default_speed = data['Speed'].mean()
        default_energy = data['Energy Consumption'].mean()
    
    st.info(f"Using default Speed: {default_speed:.1f} km/h and default Energy: {default_energy:.1f} kWh based on historical data.")
    
    is_weekend = 1 if date.weekday() >= 5 else 0
    is_peak = 1 if hour in list(range(7, 10)) + list(range(17, 20)) else 0
    is_holiday = 1 if date in manual_holidays else 0
    tod_str = get_time_of_day(hour)
    
    input_df = pd.DataFrame([{
        "City": city,
        "Vehicle Type": vehicle,
        "Weather_Category": weather,
        "Economic Condition": econ,
        "Day Of Week": date.weekday(),
        "Hour Of Day": hour,
        "Speed": default_speed,
        "Is Peak Hour": is_peak,
        "Random Event Occurred": random_event,
        "Energy Consumption": default_energy,
        "Is_Public_Holiday": is_holiday,
        "Is_Weekend": is_weekend,
        "Time of Day": tod_str,
        "Speed_Traffic_Impact_Label": 2  # Neutral placeholder
    }])
    
    st.write("Model Input:", input_df)
    
    if st.button("🚀 Predict"):
        if preprocessor:
            input_transformed = preprocessor.transform(input_df)
        else:
            input_transformed = input_df.values
        pred = selected_model.predict(input_transformed)[0]
        label_list = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        pred = int(np.clip(pred.item(), 0, len(label_list) - 1))
        label = label_list[pred]
        emoji = ['🟢', '🟢', '🟡', '🟠', '🔴'][pred]
        st.success(f"Predicted Traffic: {label} {emoji}")
        

        # Rule-based departure time suggestion and additional recommendations:
        if pred >= 3:
            # ————— Calculate an estimated travel time in traffic —————
            slowdown = 0.7 if pred == 3 else 0.5  # 70% speed for “High”, 50% for “Very High”
            effective_speed = default_speed * slowdown
            est_time_min = distance / effective_speed * 60  # minutes
            st.info(f"Estimated travel time under current conditions: **{est_time_min:.0f} minutes**")
            
            # Default suggestion for general users: depart one hour earlier (if morning/afternoon/evening)
            if tod_str in ["Morning", "Afternoon", "Evening"]:
                recommended = max(0, hour - 1)
                user_message = f"Consider departing earlier. Recommended departure time: **{recommended}:00**"
            elif tod_str == "Night":
                recommended = (hour + 1) % 24
                user_message = f"Consider departing later. Recommended departure time: **{recommended}:00**"
            st.warning("Heavy traffic detected.")
            st.info(user_message)


            # Additional suggestions specifically for City Planners:
            if role == "City Planner":
                st.markdown("""
                **Recommendations for City Planners:**
                - **Review Traffic Signal Timings:** Consider adaptive signal control strategies to better distribute traffic flow.
                - **Analyze Traffic Patterns:** Use detailed dashboard analytics to identify congestion hotspots and optimize road usage.
                - **Implement Rerouting Strategies:** Investigate opportunities for alternate routes or increased public transit frequency.
                - **Infrastructure Improvements:** Evaluate whether road widening, additional lanes or other infrastructure enhancements are warranted.
                - **Promote Alternative Transit:** Encourage cycling, walking or ride-sharing initiatives to reduce overall congestion.
                """)
        else:
            st.info("Traffic is smooth. Enjoy your journey!")



# -----------------------------
# VISUALIZATIONS PAGE
# -----------------------------
elif page == "📊 Visualizations":
    st.markdown("<h1 style='text-align:center;'>📊 Custom Traffic Visualizations</h1>", unsafe_allow_html=True)

    # Visualization filters arranged in two columns
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            city    = st.selectbox("City", sorted(data['City'].unique()), key="city")
            vehicle = st.selectbox("Vehicle Type", sorted(data['Vehicle Type'].unique()), key="vehicle")
        with col2:
            weather = st.selectbox("Weather", sorted(data['Weather_Category'].unique()), key="weather")
            econ    = st.selectbox("Economic Condition", sorted(data['Economic Condition'].unique()), key="econ")

    # Apply filters to obtain the subset data. Fallback to city-level if no match is found.
    filtered = data[(data['City'] == city) &
                    (data['Vehicle Type'] == vehicle) &
                    (data['Weather_Category'] == weather)]
    if filtered.empty:
        st.warning("No data found. Showing fallback city-level data.")
        filtered = data[data['City'] == city]


    st.markdown("### Visualization Options")
    st.markdown(
    """
    <style>
    .explanation-box {
        background-color: #e0f7fa;
        color: #000000;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)


    # Show-explanations opt-in
    show_expl = st.checkbox("Show explanations", value=False, help="Check to load detailed insights for the chosen plot")

    
    # Include all six visualization options for the users to choose.
    filters = {"city": city, "vehicle": vehicle, "weather": weather, "econ": econ}
    plot_choice = st.radio("Select a visualization:", 
                        ["Hourly Impact", 
                         "Weather Impact", 
                         "Speed vs Energy", 
                         "Traffic Impact Heatmap", 
                         "Multi-City Traffic Trend"])

    

    if plot_choice == "Hourly Impact":
        st.info("""
        🕗 **Hourly Traffic Impact – Description**
        This visualization shows how average traffic congestion changes hour by hour throughout the day.

        **How to interpret:**
        - The bar chart highlights how congestion levels differ at each hour.
        - The line chart makes it easier to spot hourly trends.

        **Example:**
        If there's a spike around 8:00 AM, it likely represents morning rush hour in the selected city.
        """)

        hourly_data = filtered.groupby("Hour Of Day")["Speed_Traffic_Impact_Label"].mean().reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(hourly_data, x="Hour Of Day", y="Speed_Traffic_Impact_Label",
                             title="Hourly Traffic Impact", color="Speed_Traffic_Impact_Label", color_continuous_scale="YlOrRd")
            st.plotly_chart(fig_bar, use_container_width=True)
        with col2:
            fig_line = px.line(hourly_data, x="Hour Of Day", y="Speed_Traffic_Impact_Label",
                               title="Hourly Traffic Trend", markers=True, color_discrete_sequence=["blue"])
            st.plotly_chart(fig_line, use_container_width=True)

    elif plot_choice == "Weather Impact":
        st.info("""
        🌧️ **Weather Impact – Description**
        Shows average traffic impact under different weather conditions (e.g., Rainy, Clear).

        **How to interpret:**
        - Bar chart shows how congestion changes depending on weather.

        **Example:**
        If impact is higher on rainy days, then drivers should avoid traveling during storms when possible.
        """)

        weather_data = filtered.groupby("Weather_Category")["Speed_Traffic_Impact_Label"].mean().reset_index()
        fig_bar = px.bar(weather_data, x="Weather_Category", y="Speed_Traffic_Impact_Label",
                         title="Weather Impact on Traffic", color="Speed_Traffic_Impact_Label", color_continuous_scale="PuBuGn")
        #st.plotly_chart(fig_bar, use_container_width=True)
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.plotly_chart(fig_bar, use_container_width=True)


    elif plot_choice == "Speed vs Energy":
        st.info("""
        ⚡ **Speed vs Energy – Description**
        Illustrates how vehicle speed relates to energy consumption and congestion severity.

        **How to interpret:**
        - Each dot shows a data point: color = congestion level.
        - A trendline helps you observe the general pattern.

        **Example:**
        If energy use increases as speed rises, slower speeds may be more efficient. A green dot with high speed means smooth flow.
        """)

        fig_trend = px.scatter(filtered, x="Speed", y="Energy Consumption", color="Speed_Traffic_Impact_Label",
                               trendline="ols", title="Speed vs Energy (with Trendline)", color_continuous_scale="Viridis")
        #st.plotly_chart(fig_trend, use_container_width=True)
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.plotly_chart(fig_trend, use_container_width=True)


    elif plot_choice == "Traffic Impact Heatmap":
        st.info("""
        🔥 **Traffic Impact Heatmap – Description**
        Provides a day-vs-hour matrix showing average congestion.

        **How to interpret:**
        - Brighter areas = higher congestion.
        - Rows = days of week, columns = hours of day.

        **Example:**
        If red zones appear on weekdays at 5 PM, that’s evening rush hour.
        """)

        pivot_data = data.pivot_table(index="Day Of Week", columns="Hour Of Day", values="Speed_Traffic_Impact_Label", aggfunc="mean")
        fig_heat = px.imshow(pivot_data,
                             labels=dict(x="Hour Of Day", y="Day Of Week", color="Avg Traffic Impact"),
                             x=pivot_data.columns,
                             y=pivot_data.index,
                             title="Traffic Impact Heatmap (Day vs Hour)",
                             color_continuous_scale="Inferno")
        #st.plotly_chart(fig_heat, use_container_width=True)
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.plotly_chart(fig_heat, use_container_width=True)


    elif plot_choice == "Multi-City Traffic Trend":
        st.info("""
        🏙️ **Multi-City Traffic Trend – Description**
        Compares traffic impact trends across cities over the course of the day.

        **How to interpret:**
        - Each line = a city’s congestion trend over 24 hours.

        **Example:**
        If City A shows lower congestion than City B at 12 PM, then City A may have smoother midday traffic.
        """)

        city_trend = data.groupby(["City", "Hour Of Day"])["Speed_Traffic_Impact_Label"].mean().reset_index()
        fig_city = px.line(city_trend, x="Hour Of Day", y="Speed_Traffic_Impact_Label", color="City",
                           title="Traffic Impact Trends by City",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        #st.plotly_chart(fig_city, use_container_width=True)
        left, center, right = st.columns([1, 4, 1])
        with center:
            st.plotly_chart(fig_city, use_container_width=True)


    # Explanation of label meaning (shown once below all charts)
    st.markdown("### About `Speed_Traffic_Impact_Label`")
    st.markdown("""
    This numeric value indicates the predicted traffic congestion level based on average speed:

    - `0` = Very Low  
    - `1` = Low  
    - `2` = Medium  
    - `3` = High  
    - `4` = Very High  

    **Note:** Higher values mean heavier traffic congestion. This label helps users understand how speed influences overall traffic flow.
    """)

    if show_expl:
        # with st.spinner("Loading explanation..."):
        #     hybrid_explanation = generate_hybrid_explanation(plot_choice, filters)
        # st.markdown(hybrid_explanation, unsafe_allow_html=True)
        with st.spinner("Loading explanation…"):
            explanation = generate_explanation(plot_choice, filters)
        st.markdown(
            f"<div class='explanation-box'>{explanation}</div>",
            unsafe_allow_html=True
        )




# -----------------------------
# REPORTS PAGE
# -----------------------------
elif page == "📥 Reports":
    st.markdown("<h1 style='text-align: center;'>📥 Generate Traffic Reports</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        city = st.selectbox("City", sorted(data['City'].unique()), key="city")
        hour = st.slider("Hour Of Day", 0, 23, key="hour")
    with col2:
        weather = st.selectbox("Weather", sorted(data['Weather_Category'].unique()), key="weather")
        vehicle = st.selectbox("Vehicle Type", sorted(data['Vehicle Type'].unique()), key="vehicle")
        econ    = st.selectbox("Economic Condition", sorted(data['Economic Condition'].unique()), key="econ")


    # Let the user choose which visualizations to include in the report
    vis_options = ["Hourly Impact", "Weather Impact", "Speed vs Energy", "Impact Histogram",
                   "Traffic Impact Heatmap", "Multi-City Traffic Trend"]
    selected_vis = st.multiselect("Choose Visualizations for Report", vis_options, default=vis_options)
    
    filtered = data[(data['City'] == city) & (data['Hour Of Day'] == hour) &
                    (data['Weather_Category'] == weather) & (data['Vehicle Type'] == vehicle) &
                    (data['Economic Condition'] == econ)]
    if filtered.empty:
        st.warning("No data found. Showing fallback city-level data.")
        filtered = data[data['City'] == city]
    
    # Show data table and CSV download
    st.dataframe(filtered.describe().reset_index())
    csv = filtered.to_csv(index=False)
    st.download_button("📄 Download CSV", csv, f"traffic_data_{city}.csv", mime="text/csv")
    

    # Generate PDF report
    if st.button("📄 Generate PDF Report"):
        filename = f"report_{city}.pdf"
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Header
        pdf.set_font("Times", "B", 16)
        pdf.cell(0, 10, safe_pdf_text(f"Smart City Traffic Report - {city}"), ln=True, align='C')
        pdf.set_font("Times", "", 12)
        pdf.cell(0, 10, safe_pdf_text(f"Report Date: {datetime.date.today()}"), ln=True, align='C')
        pdf.ln(5)
        
        
        # Executive Summary
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, safe_pdf_text("Executive Summary"), ln=True)
        pdf.set_font("Times", "", 12)
        summary_text = (
            "This report provides an overview of the current traffic conditions for the selected city. "
            "It includes key metrics such as average speed, energy consumption, and traffic impact levels. "
            "The report is intended to help urban planners and traffic management professionals make informed decisions."
        )
        pdf.multi_cell(0, 10, safe_pdf_text(summary_text))
        pdf.ln(5)
        
        # Key Metrics
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, safe_pdf_text("Key Metrics"), ln=True)
        pdf.set_font("Times", "", 12)
        metrics = {
            "Average Speed": f"{filtered['Speed'].mean():.2f} km/h",
            "Average Energy Consumption": f"{filtered['Energy Consumption'].mean():.2f} kWh",
            "Average Traffic Impact": f"{filtered['Speed_Traffic_Impact_Label'].mean():.2f}"
        }
        for key, value in metrics.items():
            pdf.cell(0, 10, safe_pdf_text(f"{key}: {value}"), ln=True)
        pdf.ln(5)
        
        # Traffic Prediction & Suggestions
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Traffic Prediction & Suggestions", ln=True)
        pdf.set_font("Times", "", 12)
        avg_impact = filtered['Speed_Traffic_Impact_Label'].mean()
        if avg_impact < 1:
            pred_label = "Very Low"
            suggestion = "Traffic is very light. No major delays expected."
        elif avg_impact < 2:
            pred_label = "Low"
            suggestion = "Traffic is light. Minimal congestion observed."
        elif avg_impact < 3:
            pred_label = "Medium"
            suggestion = "Traffic is moderate. Stay alert and plan accordingly."
        elif avg_impact < 4:
            pred_label = "High"
            suggestion = "Heavy traffic detected. Consider adjusting your departure time."
        else:
            pred_label = "Very High"
            suggestion = "Traffic is extremely congested. Consider delaying your trip."
        pdf.cell(0, 10, f"Predicted Traffic Level: {pred_label}", ln=True)
        pdf.multi_cell(0, 10, f"Suggestion: {suggestion}")
        pdf.ln(5)
        
        # Visualizations Section – include all selected plots
        pdf.set_font("Times", "B", 14)
        pdf.cell(0, 10, "Visualizations & Explanations", ln=True)
        pdf.ln(3)
        
        for vis in selected_vis:
            pdf.set_font("Times", "B", 14)
            pdf.cell(0, 10, f"{vis}:", ln=True)
            
            # Generate the corresponding plot for each chosen visualization
            vis_fig_path = f"{vis}.png"
            if vis == "Hourly Impact":
                hourly_data = filtered.groupby("Hour Of Day")["Speed_Traffic_Impact_Label"].mean().reset_index()
                fig, ax = plt.subplots()
                ax.bar(hourly_data["Hour Of Day"], hourly_data["Speed_Traffic_Impact_Label"], color="blue")
                ax.set_title("Hourly Traffic Impact")
                ax.set_xlabel("Hour Of Day")
                ax.set_ylabel("Avg Traffic Impact")
                fig.savefig(vis_fig_path)
                plt.close(fig)
            elif vis == "Weather Impact":
                weather_data = filtered.groupby("Weather_Category")["Speed_Traffic_Impact_Label"].mean().reset_index()
                fig, ax = plt.subplots()
                ax.bar(weather_data["Weather_Category"], weather_data["Speed_Traffic_Impact_Label"], color="orange")
                ax.set_title("Weather Impact on Traffic")
                ax.set_xlabel("Weather Category")
                ax.set_ylabel("Avg Traffic Impact")
                fig.savefig(vis_fig_path)
                plt.close(fig)
            elif vis == "Speed vs Energy":
                fig, ax = plt.subplots()
                ax.scatter(filtered["Speed"], filtered["Energy Consumption"], color="green")
                ax.set_title("Speed vs Energy with Traffic Impact")
                ax.set_xlabel("Speed (km/h)")
                ax.set_ylabel("Energy Consumption (kWh)")
                fig.savefig(vis_fig_path)
                plt.close(fig)
            elif vis == "Impact Histogram":
                fig, ax = plt.subplots()
                ax.hist(filtered["Speed_Traffic_Impact_Label"], bins=5, color="#636EFA")
                ax.set_title("Traffic Impact Histogram")
                ax.set_xlabel("Traffic Impact Level")
                ax.set_ylabel("Frequency")
                fig.savefig(vis_fig_path)
                plt.close(fig)
            elif vis == "Traffic Impact Heatmap":
                pivot_data = data.pivot_table(index="Day Of Week", columns="Hour Of Day",
                                              values="Speed_Traffic_Impact_Label", aggfunc="mean")
                fig, ax = plt.subplots()
                cax = ax.imshow(pivot_data, aspect="auto", cmap="viridis")
                ax.set_title("Traffic Impact Heatmap (Day vs Hour)")
                ax.set_xlabel("Hour Of Day")
                ax.set_ylabel("Day Of Week")
                fig.colorbar(cax)
                fig.savefig(vis_fig_path)
                plt.close(fig)
            elif vis == "Multi-City Traffic Trend":
                city_trend = data.groupby(["City", "Hour Of Day"])["Speed_Traffic_Impact_Label"].mean().reset_index()
                fig, ax = plt.subplots()
                for key, grp in city_trend.groupby("City"):
                    ax.plot(grp["Hour Of Day"], grp["Speed_Traffic_Impact_Label"], label=key)
                ax.set_title("Traffic Impact Trends by City")
                ax.set_xlabel("Hour Of Day")
                ax.set_ylabel("Avg Traffic Impact")
                ax.legend()
                fig.savefig(vis_fig_path)
                plt.close(fig)
            else:
                vis_fig_path = None
            
            if vis_fig_path and os.path.exists(vis_fig_path):
                pdf.image(vis_fig_path, x=10, w=pdf.w - 20)
                os.remove(vis_fig_path)
                pdf.ln(3)
            
            # Include the explanation for the selected visualization.
            pdf.set_font("Times", "", 12)
            vis_explanation_html = generate_hybrid_explanation(vis, {"city": city, "vehicle": vehicle, "weather": weather, "econ": econ})
            vis_explanation_plain = strip_html(vis_explanation_html)
            pdf.multi_cell(0, 10, vis_explanation_plain)
            pdf.ln(5)
        
        # Footer
        pdf.set_font("Times", "", 10)
        pdf.cell(0, 10, "This report was generated by the Smart City Traffic System.", ln=True, align='C')
        
        pdf_bytes = pdf.output(dest="S").encode("latin1", errors="replace")
        st.download_button("📥 Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")


# -----------------------------
# FEEDBACK PAGE
# -----------------------------
elif page == "💬 Feedback":
    st.markdown("<h1 style='text-align:center;'>💬 User Feedback</h1>", unsafe_allow_html=True)

    feedback_file = "feedback.csv"
    if role == "Admin":
        st.markdown("### Feedback Received")
        if os.path.exists(feedback_file):
            # Use 'on_bad_lines' to skip rows with extra fields
            feedback_df = pd.read_csv(feedback_file, header=None, 
                                      names=["Timestamp", "Agreement", "Feedback"],
                                      on_bad_lines='skip')
            feedback_df = feedback_df.reset_index().rename(columns={"index": "ID"})
            st.dataframe(feedback_df)
            ids_to_delete = st.multiselect("Select Feedback IDs to Delete", options=feedback_df["ID"].tolist())
            if st.button("Delete Selected Feedback"):
                if len(ids_to_delete) > 0:
                    feedback_df = feedback_df[~feedback_df["ID"].isin(ids_to_delete)]
                    feedback_df.drop(columns=["ID"], inplace=True)
                    feedback_df.to_csv(feedback_file, index=False, header=False)
                    st.success("Selected feedback entries have been deleted. Please refresh the page to see updates.")
                else:
                    st.info("No feedback entries selected for deletion.")
        else:
            st.info("No feedback available yet.")
    else:
        st.markdown("Please share your thoughts on the traffic predictions:")
        with st.form("feedback_form"):
            agree = st.radio("Do you agree with the prediction?", ["Yes", "No"])
            actual_feedback = st.text_input("What was the actual traffic like for you?")
            submit_feedback = st.form_submit_button("Submit Feedback")
            if submit_feedback:
                st.success("Thank you for your feedback!")
                with open(feedback_file, "a") as f:
                    f.write(f"{datetime.datetime.now()},{agree},{actual_feedback}\n")


    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image("C:/Users/Ning Sheng Yong/Desktop/QING APU/SystemImages/9583344.gif", use_container_width=True)


# -----------------------------
# ABOUT PAGE
# -----------------------------   
elif page == "ℹ️ About":
    st.markdown("<h1 style='text-align:center;'>About Smart City Traffic System</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("C:/Users/Ning Sheng Yong/Desktop/QING APU/SystemImages/1709808629051.png", width=750)
    with col2:
        st.image("C:/Users/Ning Sheng Yong/Desktop/QING APU/SystemImages/1700487546341.png", width=500)


    st.markdown("""
    ### Overview  
    The **Smart City Traffic System** provides a comprehensive dashboard for urban traffic monitoring and prediction using artificial intelligence.

    Built using futuristic traffic datasets, the system offers insights such as:
    - Futuristic Traffic Predictions
    - In-depth Trend Visualizations
    - Custom Reports for Policy and Planning
    - AI-Generated Explanations for Non-Professional Users

    ### Features
    - **User-Friendly Interface** with customizable filters.
    - **Role-Based Access**: General Users, City Planners and Admins.
    - **Interactive Visualizations** tailored to traffic patterns.
    - **Departure Time Advice** with predicted congestion levels.
    - **Downloadable PDF/CSV Reports** for analytics or planning use.
    - **Feedback System** to continuously improve predictions.

    ### Why CatBoost Was Chosen  
    The deployed model behind the predictions is **CatBoost**, a cutting-edge gradient boosting algorithm developed by Yandex.

    **Justification for Choosing CatBoost:**
    - **High Accuracy**: CatBoost achieved over **82% accuracy** across five traffic impact levels, outperforming other models like Logistic Regression and Random Forest.
    - **Excellent with Categorical Data**: Traffic data includes features like city name, vehicle type and weather condition. CatBoost can handle categorical variables without requiring manual encoding, improving both accuracy and speed.
    - **Fast and Efficient**: It is highly optimized for performance on tabular datasets and performs well even with imbalanced classes when used with SMOTE.
    - **Great for Small to Medium Datasets**: Unlike deep learning models, CatBoost doesn’t require millions of records or GPUs, making it ideal for realistic city datasets with limited data.

    ### How It Works
    CatBoost builds an ensemble of decision trees in sequence. Each tree tries to correct the mistakes of the previous one, resulting in highly accurate and generalizable predictions. In this project, it was trained to classify congestion into five levels: Very Low, Low, Medium, High and Very High.

    ### Why It’s Helpful
    This model enables the system to:
    - Notify users of upcoming congestion.
    - Suggest departure times during high congestion hours.
    - Help planners analyze peak hours and hotspots.
    - Support Sustainable Development Goal 11 by promoting efficient, sustainable urban mobility.

    Whether you are a user planning your trip, or a city planner making infrastructure decisions, the Smart City Traffic System powered by **CatBoost** offers reliable, data-driven assistance for a better urban experience.
    """, unsafe_allow_html=True)
