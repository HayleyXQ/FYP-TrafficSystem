

import joblib
import pandas as pd
import streamlit as st
import re
from google import genai
import streamlit as st


@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    data = pd.read_csv(path)
    
    # Map traffic density category to numeric code if needed
    if 'Density_Code' not in data.columns and 'Traffic Density Category' in data.columns:
        density_map = {
            'very low': 0,
            'low': 1,
            'medium': 2,
            'high': 3,
            'very high': 4
        }
        data['Density_Code'] = data['Traffic Density Category'].map(density_map)
    
    return data

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
    elif 'rain' in weather:
        return 'Rainy'
    elif 'snow' in weather:
        return 'Snowy'
    else:
        return 'Other'

def mappings(data):
    return {
        'city': {city: i for i, city in enumerate(sorted(data['City'].unique()))},
        'vehicle': {v: i for i, v in enumerate(sorted(data['Vehicle Type'].unique()))},
        'weather': {w: i for i, w in enumerate(sorted(data['Weather'].unique()))},
        'economic': {e: i for i, e in enumerate(sorted(data['Economic Condition'].unique()))},
        'time_of_day': {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3},
        'weather_category': {'Clear': 0, 'Rainy': 1, 'Snowy': 2, 'Other': 3}
    }



GEMINI_API_KEY = "AIzaSyAF_dpw5T3INCM-y9NNZW8z9uaEbdizwps"
client = genai.Client(api_key=GEMINI_API_KEY)


def gemini_generate_explanation(
    prompt: str,
    model_name: str = "gemini-2.0-flash",
    max_tokens: int = 250,
    temperature: float = 0.5
) -> str:
    """
    Call Google Gemini to generate a short explanation for a prompt.
    """
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    return resp.text or "No explanation returned."

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
