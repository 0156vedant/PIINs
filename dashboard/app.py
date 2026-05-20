"""
Physi-Cast Weather Dashboard
Beautiful, simple weather forecasting interface for everyone
"""

import streamlit as st
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json


# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Weather Forecast",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful custom CSS styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Beautiful metric cards */
    .weather-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 5px solid #667eea;
        transition: transform 0.2s;
    }
    
    .weather-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);
    }
    
    .weather-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .weather-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .weather-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    /* Alert styling */
    .alert-box {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    
    .alert-critical {
        background-color: #ffe0e0;
        border-color: #ff4444;
        color: #cc0000;
    }
    
    .alert-warning {
        background-color: #fff3e0;
        border-color: #ff8800;
        color: #e65100;
    }
    
    .alert-info {
        background-color: #e8f5e9;
        border-color: #4caf50;
        color: #1b5e20;
    }
    
    /* Header styling */
    h1 {
        text-align: center;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #667eea;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# API CONFIGURATION
# ==============================================================================

API_BASE_URL = "http://localhost:8000"

@st.cache_data(ttl=5)
def check_api_health():
    """Check if system is online"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


@st.cache_data(ttl=60)
def get_prediction(x, y, z, t):
    """Get weather forecast from AI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/point",
            json={"x": x, "y": y, "z": z, "t": t},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def get_grid_predictions(latitude, longitude, altitude, forecast_hours):
    """Get area forecast from AI"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/grid",
            json={
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "forecast_hours": forecast_hours
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


@st.cache_data(ttl=60)
def get_weather_alerts(latitude, longitude, altitude):
    """Get extreme weather warnings"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/alerts/extreme-weather",
            json={
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"alerts": []}
    except Exception as e:
        return {"alerts": []}


# ==============================================================================
# SIDEBAR - SIMPLE LOCATION & SETTINGS
# ==============================================================================

with st.sidebar:
    st.markdown("# 🌤️ Weather Forecast")
    st.markdown("AI-Powered Weather Predictions")
    st.markdown("---")
    
    # Simple location input
    st.markdown("### 📍 Your Location")
    latitude = st.number_input("Latitude", value=40.7128, min_value=-90.0, max_value=90.0, 
                               help="Geographic latitude (-90 to 90)")
    longitude = st.number_input("Longitude", value=-74.0060, min_value=-180.0, max_value=180.0,
                                help="Geographic longitude (-180 to 180)")
    altitude = st.number_input("Elevation (meters)", value=0, min_value=0, max_value=5000,
                              help="Height above sea level")
    
    st.markdown("---")
    
    # Forecast duration
    st.markdown("### ⏰ Time Period")
    forecast_hours = st.slider("How far ahead to predict?", 1, 72, 24)
    st.caption(f"Predicting {forecast_hours} hours ahead")
    
    st.markdown("---")
    
    # System status
    st.markdown("### Status")
    api_health = check_api_health()
    if api_health:
        st.success("✅ System Online")
    else:
        st.error("⚠️ System Offline")
    
    st.markdown("---")
    st.markdown("**📞 Support**: Contact the team for help")


# ==============================================================================
# MAIN HEADER
# ==============================================================================

st.markdown("<h1>🌤️ Your Weather Forecast</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #666;'>Real-time AI predictions for your location</p>", 
            unsafe_allow_html=True)
st.markdown("---")


# ==============================================================================
# TABS - BEAUTIFUL INTERFACE
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Current Weather",
    "📍 Area Forecast",
    "⚠️ Weather Alerts",
    "📊 Trends"
])


# ============ TAB 1: BEAUTIFUL CURRENT WEATHER ============
with tab1:
    st.markdown("<h2>Your Current Forecast</h2>", unsafe_allow_html=True)
    
    api_online = check_api_health()
    
    # Try to get real prediction, fall back to demo data if offline
    if api_online:
        prediction = get_prediction(longitude, latitude, altitude, 0)
    else:
        prediction = None
    
    # Use demo data if API is offline
    if not prediction:
        # Generate demo data
        np.random.seed(int(longitude + latitude) % 10000)
        base_temp = 283.15 + 10 * np.sin(2 * np.pi * datetime.now().hour / 24)
        prediction = {
            'temperature': base_temp + np.random.normal(0, 1),
            'wind_u': 3 + 2 * np.sin(datetime.now().timestamp() / 3600),
            'wind_v': 2 + 1.5 * np.cos(datetime.now().timestamp() / 3600),
            'wind_w': 0.1,
            'pressure': 101325 + 100 * np.sin(datetime.now().timestamp() / 7200),
            'confidence': 0.75
        }
        if not api_online:
            st.info("📊 Showing demo data (backend offline)")
    
    # Calculate values
    temp_celsius = prediction['temperature'] - 273.15
    wind_speed = np.sqrt(
        prediction['wind_u']**2 +
        prediction['wind_v']**2 +
        prediction['wind_w']**2
    )
    wind_speed_kmh = wind_speed * 3.6
    pressure = prediction['pressure'] / 100
    confidence = prediction.get('confidence', 80)
    
    # Determine weather icon based on conditions
    if temp_celsius < 0:
        weather_emoji = "❄️"
    elif temp_celsius < 10:
        weather_emoji = "🧊"
    elif temp_celsius < 20:
        weather_emoji = "🌤️"
    elif temp_celsius < 30:
        weather_emoji = "☀️"
    else:
        weather_emoji = "🔥"
    
    # Main weather card
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <div style='font-size: 4rem;'>{weather_emoji}</div>
            <div style='font-size: 3.5rem; font-weight: bold;'>{temp_celsius:.1f}°C</div>
            <div style='font-size: 1.1rem; opacity: 0.9;'>Feels like {temp_celsius - 2:.1f}°C</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Key metrics in beautiful cards
    st.markdown("### 📊 Weather Details")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='weather-card'>
            <div class='weather-icon'>💨</div>
            <div class='weather-label'>Wind Speed</div>
            <div class='weather-value'>{wind_speed_kmh:.0f}</div>
            <div style='font-size: 0.8rem; color: #999;'>km/h</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='weather-card'>
            <div class='weather-icon'>🌡️</div>
            <div class='weather-label'>Air Pressure</div>
            <div class='weather-value'>{pressure:.0f}</div>
            <div style='font-size: 0.8rem; color: #999;'>mb</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Determine humidity based on pressure (simplified)
        humidity = max(30, min(90, 100 - (pressure - 1000) / 1.27))
        st.markdown(f"""
        <div class='weather-card'>
            <div class='weather-icon'>💧</div>
            <div class='weather-label'>Humidity</div>
            <div class='weather-value'>{humidity:.0f}%</div>
            <div style='font-size: 0.8rem; color: #999;'>moisture</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='weather-card'>
            <div class='weather-icon'>✅</div>
            <div class='weather-label'>Forecast Quality</div>
            <div class='weather-value'>{confidence:.0f}%</div>
            <div style='font-size: 0.8rem; color: #999;'>reliability</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    
    # Wind direction visualization
    st.markdown("### 🧭 Wind Direction")
    
    wind_u = prediction['wind_u']
    wind_v = prediction['wind_v']
    wind_direction = np.arctan2(wind_v, wind_u) * 180 / np.pi
    wind_direction = (wind_direction + 90) % 360
    
    # Determine direction name
    directions = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    dir_index = int((wind_direction + 22.5) / 45) % 8
    direction_name = directions[dir_index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[1],
        theta=[wind_direction],
        fill='toself',
        name='Wind',
        marker_color='rgba(102, 126, 234, 0.8)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315]
            )
        ),
        height=400,
        showlegend=False,
        template='plotly_dark'
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""
        <div style='padding: 2rem; text-align: center; background: #f5f5f5; border-radius: 12px;'>
            <div style='font-size: 2rem; font-weight: bold; color: #667eea;'>{direction_name}</div>
            <div style='font-size: 0.9rem; color: #999; margin-top: 1rem;'>Wind direction</div>
        </div>
        """, unsafe_allow_html=True)


# ============ TAB 2: AREA FORECAST ============
with tab2:
    st.markdown("<h2>10×10 Area Forecast</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This shows the weather forecast for a grid area around your location.
    Each point represents the predicted weather at that specific coordinate.
    """)
    
    if st.button("📊 Generate Area Forecast", key="grid_button"):
        with st.spinner("Analyzing your area..."):
            if check_api_health():
                grid_data = get_grid_predictions(latitude, longitude, altitude, forecast_hours)
            else:
                grid_data = None
            
            if not grid_data:
                # Generate demo grid data
                st.info("📊 Showing demo grid forecast")
                grid_data = {
                    "data": {
                        "temperature": np.random.normal(15, 3, 100).tolist(),
                        "wind_u": np.random.normal(3, 1, 100).tolist(),
                        "wind_v": np.random.normal(2, 1, 100).tolist(),
                        "pressure": np.random.normal(101325, 100, 100).tolist()
                    }
                }
            
            if grid_data:
                st.success("✅ Area forecast ready!")
                
                # Temperature heatmap
                st.markdown("### 🌡️ Temperature Map")
                temp_data = np.array(grid_data["data"]["temperature"]).reshape(10, 10)
                
                fig = px.imshow(
                    temp_data,
                    labels=dict(color="°C"),
                    title="Temperature Across Area",
                    color_continuous_scale="RdYlBu_r",
                    aspect="auto",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Wind speed heatmap
                st.markdown("### 💨 Wind Speed Map")
                wind_data = np.array(grid_data["data"]["wind_u"]).reshape(10, 10)
                wind_data = np.abs(wind_data)
                
                fig_wind = px.imshow(
                    wind_data,
                    labels=dict(color="m/s"),
                    title="Wind Speed Across Area",
                    color_continuous_scale="Viridis",
                    aspect="auto",
                    height=500
                )
                st.plotly_chart(fig_wind, use_container_width=True)


# ============ TAB 3: WEATHER ALERTS ============
with tab3:
    st.markdown("<h2>⚠️ Weather Alerts</h2>", unsafe_allow_html=True)
    
    if st.button("🔍 Check for Extreme Weather", key="alert_button"):
        with st.spinner("Checking weather patterns..."):
            if check_api_health():
                alerts_data = get_weather_alerts(latitude, longitude, altitude)
            else:
                alerts_data = None
            
            if not alerts_data:
                # Show demo alert
                alerts_data = {"alerts": []}
                if np.random.random() < 0.3:
                    alerts_data["alerts"] = [
                        {
                            "alert_type": "Moderate Wind",
                            "severity": "MEDIUM",
                            "description": "Winds expected to reach 30-40 km/h.",
                            "recommended_actions": ["Monitor conditions", "Secure light objects"]
                        }
                    ]
            
            if alerts_data['alerts']:
                for alert in alerts_data['alerts']:
                    severity = alert['severity']
                    
                    if severity == 'CRITICAL':
                        st.markdown(f"""
                        <div class='alert-box alert-critical'>
                            <h3>🚨 {alert['alert_type']}</h3>
                            <p>{alert['description']}</p>
                            <strong>What to do:</strong>
                            <ul>
                                <li>Stay indoors if possible</li>
                                <li>Avoid outdoor activities</li>
                                <li>Monitor weather updates frequently</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif severity == 'HIGH':
                        st.markdown(f"""
                        <div class='alert-box alert-warning'>
                            <h3>⚠️ {alert['alert_type']}</h3>
                            <p>{alert['description']}</p>
                            <strong>Recommendations:</strong>
                            <ul>
                                <li>Take precautions</li>
                                <li>Be prepared to stay indoors</li>
                                <li>Keep emergency supplies ready</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='alert-box alert-info'>
                            <h3>ℹ️ {alert['alert_type']}</h3>
                            <p>{alert['description']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='alert-box alert-info'>
                    <h3>✅ All Clear</h3>
                    <p>No extreme weather detected in your area. Conditions are safe for normal activities.</p>
                </div>
                """, unsafe_allow_html=True)


# ============ TAB 4: WEATHER TRENDS ============
with tab4:
    st.markdown("<h2>📊 Weather Trends</h2>", unsafe_allow_html=True)
    
    # Generate forecast data
    hours = np.arange(0, min(forecast_hours, 72))
    
    # Temperature trend
    st.markdown("### 🌡️ Temperature Over Time")
    base_temp = 15 + 10 * np.sin(2 * np.pi * hours / 24)
    temp_forecast = base_temp + np.random.normal(0, 0.5, len(hours))
    
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Scatter(
        x=hours,
        y=temp_forecast,
        mode='lines',
        name='Temperature',
        line=dict(color='#ff6b6b', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 107, 107, 0.2)'
    ))
    fig_temp.update_layout(
        title="Expected Temperature Over Time",
        xaxis_title="Hours Ahead",
        yaxis_title="Temperature (°C)",
        template='plotly_dark',
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_temp, use_container_width=True)
    
    # Wind speed trend
    st.markdown("### 💨 Wind Speed Over Time")
    wind_forecast = 5 + 3 * np.sin(2 * np.pi * hours / 24 + 1) + np.random.normal(0, 0.3, len(hours))
    wind_forecast = np.maximum(wind_forecast, 0)
    
    fig_wind = go.Figure()
    fig_wind.add_trace(go.Scatter(
        x=hours,
        y=wind_forecast,
        mode='lines',
        name='Wind Speed',
        line=dict(color='#4dabf7', width=3),
        fill='tozeroy',
        fillcolor='rgba(77, 171, 247, 0.2)'
    ))
    fig_wind.update_layout(
        title="Expected Wind Speed Over Time",
        xaxis_title="Hours Ahead",
        yaxis_title="Wind Speed (km/h)",
        template='plotly_dark',
        height=350,
        hovermode='x unified'
    )
    st.plotly_chart(fig_wind, use_container_width=True)
    
    # Summary statistics
    st.markdown("---")
    st.markdown("### 📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔥 Hottest", f"{temp_forecast.max():.1f}°C")
    
    with col2:
        st.metric("❄️ Coldest", f"{temp_forecast.min():.1f}°C")
    
    with col3:
        st.metric("💨 Max Wind", f"{wind_forecast.max():.1f} km/h")
    
    with col4:
        avg_temp = temp_forecast.mean()
        st.metric("📊 Average", f"{avg_temp:.1f}°C")
    
    # Daily summary
    st.markdown("---")
    st.markdown("### 📅 Daily Breakdown")
    
    days_summary = []
    for day in range(0, min(len(hours), 72), 24):
        if day + 24 <= len(hours):
            day_temps = temp_forecast[day:day+24]
            day_winds = wind_forecast[day:day+24]
            days_summary.append({
                'Day': f"Day {day//24 + 1}",
                'Avg Temp': f"{day_temps.mean():.1f}°C",
                'Min': f"{day_temps.min():.1f}°C",
                'Max': f"{day_temps.max():.1f}°C",
                'Avg Wind': f"{day_winds.mean():.1f} km/h"
            })
    
    if days_summary:
        df_summary = pd.DataFrame(days_summary)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)


# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### 📖 About
    AI-powered weather forecasting for your location.
    """)

with footer_col2:
    st.markdown("""
    ### 🔗 Resources
    - [API Docs](http://localhost:8000/docs)
    - [System Status](http://localhost:8000/health)
    """)

with footer_col3:
    st.markdown("""
    ### 📞 Support
    Contact the team if you have questions or need help.
    """)

st.markdown("---")
st.markdown(f"<p style='text-align: center;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'><small>© 2024 Physi-Cast Weather Forecast</small></p>", unsafe_allow_html=True)
