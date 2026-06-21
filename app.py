import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Add src/ to system path so we can import backend packages directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from backend.data_cleaning import clean_dataset, prepare_features
from backend.feature_engineering import create_features
from backend.prediction import load_model, predict_price
from backend.database import create_database, save_prediction, load_history

# Set premium page layout
st.set_page_config(
    page_title="CarValue v2 | Premium Valuation System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom styling template
def load_css():
    css_path = os.path.join('src', 'frontend', 'templates', 'custom_styles.css')
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load HTML component template
def load_html_component(component_name):
    html_path = os.path.join('src', 'frontend', 'templates', 'ui_components.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
            # Split the templates by comment banners
            components = html.split('<!--')
            for comp in components:
                if component_name in comp:
                    # Return the content after the comment block close
                    parts = comp.split('-->')
                    if len(parts) > 1:
                        return parts[1].strip()
    return ""

# Initialize database
create_database()

# Load custom CSS
load_css()

# Load Dataset for dynamic options (cached for high performance)
@st.cache_data
def get_dataset_info():
    dataset_path = 'used_cars_dataset_v2.csv'
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        # Populate dynamic brand and model lists
        brand_model_dict = {}
        for brand in sorted(df['Brand'].dropna().unique()):
            models = df[df['Brand'] == brand]['model'].dropna().unique()
            brand_model_dict[brand] = sorted(list(models))
        
        # Calculate summary statistics for metric cards
        total_cars = len(df)
        avg_price = 785000.0  # fallback estimation base
        if 'AskPrice' in df.columns:
            # Quick clean to find mean
            prices = df['AskPrice'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
            prices = pd.to_numeric(prices, errors='coerce').dropna()
            if len(prices) > 0:
                avg_price = prices.mean()
        
        return brand_model_dict, total_cars, avg_price, df
    else:
        # Fallback values if dataset is missing
        return {"Maruti Suzuki": ["Swift"]}, 1, 500000.0, pd.DataFrame()

brand_model_dict, total_cars_count, average_market_price, raw_df = get_dataset_info()

# Load Model Pipeline
@st.cache_resource
def get_prediction_model():
    model_path = os.path.join('models', 'car_price_model_v2.joblib')
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = get_prediction_model()

# Header Template
header_html = load_html_component("CUSTOM APP HEADER COMPONENT")
if header_html:
    st.markdown(
        header_html.format(
            header_text="🚗 CarValue v2 AI Estimator",
            subheader_text="Industry-level machine learning model to estimate the current market value of used cars in India"
        ),
        unsafe_allow_html=True
    )
else:
    st.title("🚗 CarValue v2 AI Estimator")

# Metrics Container
metrics_html = load_html_component("METRICS DISPLAY CONTAINER")
if metrics_html:
    st.markdown(
        metrics_html.format(
            m1_title="Models Checked",
            m1_value="CatBoost",
            m2_title="Database Size",
            m2_value=f"{total_cars_count:,} Cars",
            m3_title="Average Market Price",
            m3_value=f"₹{average_market_price:,.0f}"
        ),
        unsafe_allow_html=True
    )

# Create layout (Valuation input form vs Visualizations/History)
col_input, col_viz = st.columns([1.1, 1.4])

# Define segment lists (must match feature engineering file)
PREMIUM_BRANDS = ['Honda', 'Toyota', 'Volkswagen', 'Skoda']
LUXURY_BRANDS = ['Mercedes-Benz', 'BMW', 'Audi', 'Volvo', 'Jaguar', 'Land Rover']

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Car Valuation Details")
    
    # Brand select with dynamic model list
    brand = st.selectbox("Vehicle Brand", list(brand_model_dict.keys()), index=0)
    models_available = brand_model_dict.get(brand, [])
    model_name = st.selectbox("Vehicle Model", models_available, index=0)
    
    # Layout details form
    sub_col1, sub_col2 = st.columns(2)
    
    with sub_col1:
        year = st.number_input(
            "Manufacturing Year",
            min_value=1990,
            max_value=date.today().year,
            value=2018
        )
        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"]
        )
        owner_type = st.selectbox(
            "Owner Type",
            ["First", "Second", "Third", "Fourth & Above"]
        )
        
    with sub_col2:
        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=5000
        )
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "LPG"]
        )
        
    submit = st.button("💰 Calculate Value", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_viz:
    if submit:
        if model is None:
            st.error("Model file `models/car_price_model_v2.joblib` not found. Please run main.py first.")
        else:
            # 1. Feature Calculations (Workflow: cleaning + feature engineering)
            car_age = date.today().year - year
            brand_model = f"{brand}_{model_name}"
            usage_per_year = km_driven / (car_age + 1)
            is_premium = 1 if brand in PREMIUM_BRANDS else 0
            is_luxury = 1 if brand in LUXURY_BRANDS else 0
            
            # Map owner score
            owner_map = {'First': 1, 'Second': 2, 'Third': 3, 'Fourth & Above': 4}
            ownership_score = owner_map.get(owner_type, 1)
            
            # 2. Package raw input into DataFrame
            input_raw = pd.DataFrame({
                'Brand': [brand],
                'model': [model_name],
                'Year': [year],
                'kmDriven': [f"{km_driven} km"],
                'Transmission': [transmission],
                'Owner': [owner_type],
                'FuelType': [fuel_type]
            })
            
            # 3. Step 4: Perform data cleaning
            input_cleaned = clean_dataset(input_raw)
            
            # 4. Step 5: Perform feature engineering
            input_engineered = create_features(input_cleaned)
            input_prepared = prepare_features(input_engineered)
            
            # 5. Step 6: Prediction model runs
            prediction = predict_price(model, input_prepared)
            
            # Check for negative predictions (fallback safeguard)
            if prediction < 10000:
                prediction = 10000.0
                
            # Confidence logic based on age and variance (MAPE ~ 12%)
            confidence_level = max(65.0, 95.0 - (car_age * 1.8))
            deviation = prediction * 0.12 # 12% margin
            min_price = max(10000.0, prediction - deviation)
            max_price = prediction + deviation
            
            # 6. Step 7: Prediction stored in database
            save_prediction(
                brand=brand,
                model=model_name,
                year=year,
                km_driven=km_driven,
                transmission=transmission,
                fuel_type=fuel_type,
                owner_type=owner_type,
                car_age=car_age,
                predicted_price=prediction,
                confidence=confidence_level
            )
            
            # 7. Step 8: Result displayed in professional UI
            price_card_html = load_html_component("PREDICTION DETAILS DISPLAY CARD")
            if price_card_html:
                st.markdown(
                    price_card_html.format(
                        title="Estimated Market Valuation",
                        price=f"₹ {prediction:,.2f}",
                        min_price=f"₹ {min_price:,.2f}",
                        max_price=f"₹ {max_price:,.2f}",
                        confidence=f"{confidence_level:.1f}"
                    ),
                    unsafe_allow_html=True
                )
            else:
                st.success(f"### Estimated Price: ₹ {prediction:,.2f}")
                st.info(f"Range: ₹ {min_price:,.2f} - ₹ {max_price:,.2f}")
                
            # Add segment notification
            segment_label = 'Luxury' if is_luxury else 'Premium' if is_premium else 'Standard'
            st.info(f"ℹ️ **Vehicle Segment:** {segment_label} | **Estimated Annual Usage:** {usage_per_year:,.0f} km/year")
            
            # Similar Cars comparison from training data
            if not raw_df.empty:
                similar_cars = raw_df[(raw_df['Brand'] == brand) & (raw_df['model'] == model_name)].copy()
                if len(similar_cars) > 2:
                    st.subheader(f"Market Analytics: Similar {brand} {model_name} Listings")
                    
                    # Clean AskPrice for visualization
                    similar_cars['CleanPrice'] = similar_cars['AskPrice'].astype(str).str.replace('₹', '').str.replace(',', '').str.strip()
                    similar_cars['CleanPrice'] = pd.to_numeric(similar_cars['CleanPrice'], errors='coerce')
                    similar_cars['CleanKM'] = similar_cars['kmDriven'].astype(str).str.replace('km', '').str.replace(',', '').str.strip()
                    similar_cars['CleanKM'] = pd.to_numeric(similar_cars['CleanKM'], errors='coerce')
                    similar_cars = similar_cars.dropna(subset=['CleanPrice', 'CleanKM'])
                    
                    if len(similar_cars) > 1:
                        # Draw scatter plot: Price vs KM
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.scatterplot(
                            x='CleanKM', y='CleanPrice', data=similar_cars, 
                            ax=ax, hue='Year', palette='viridis', size='Year', sizes=(50, 200)
                        )
                        # Mark current prediction on plot
                        ax.scatter([km_driven], [prediction], color='red', marker='X', s=300, label='Predicted Valuation')
                        ax.set_title(f'Price vs Mileage Distribution for {brand} {model_name}')
                        ax.set_xlabel('Kilometers Driven')
                        ax.set_ylabel('Ask Price (₹)')
                        ax.legend()
                        st.pyplot(fig)
    else:
        st.markdown('<div class="glass-card" style="text-align: center; padding: 50px 20px;">', unsafe_allow_html=True)
        st.write("### 🚗 Welcome to CarValue Predictor v2")
        st.write("Enter the vehicle details in the left form panel and click **Calculate Value** to generate an AI estimation based on recent market trends.")
        st.markdown('</div>', unsafe_allow_html=True)

# Historical predictions list
st.markdown("---")
st.subheader(" Valuation History Log (SQLite predictions.db)")
history = load_history(limit=8)

if not history.empty:
    # Rename columns for presentation
    history_display = history.copy()
    history_display.columns = [
        'ID', 'Timestamp', 'Brand', 'Model', 'Year', 'KM Driven',
        'Transmission', 'Fuel Type', 'Owner', 'Age (Yrs)', 'Estimated Price (₹)', 'Confidence (%)'
    ]
    st.dataframe(
        history_display.style.format({
            'Estimated Price (₹)': '₹ {:,.2f}',
            'KM Driven': '{:,.0f} km',
            'Confidence (%)': '{:.1f}%'
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.write("No predictions logged yet. Try evaluating a vehicle to initialize the history log.")