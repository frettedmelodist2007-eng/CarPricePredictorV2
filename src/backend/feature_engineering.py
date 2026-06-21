import pandas as pd
import numpy as np
from datetime import date

# Define segments and mapping for feature engineering
PREMIUM_BRANDS = ['Honda', 'Toyota', 'Volkswagen', 'Skoda']
LUXURY_BRANDS = ['Mercedes-Benz', 'BMW', 'Audi', 'Volvo', 'Jaguar', 'Land Rover']
OWNER_MAP = {
    'first': 1,
    'second': 2,
    'third': 3,
    'fourth & above': 4
}

def create_features(df):
    """
    Creates new engineered features from standard dataset columns:
    car_age, brand_model, usage_per_year, is_premium, is_luxury, ownership_score.
    """
    df_engineered = df.copy()
    
    # 1. Car Age
    if 'car_age' not in df_engineered.columns and 'year' in df_engineered.columns:
        current_year = date.today().year
        df_engineered['car_age'] = current_year - df_engineered['year']
        
    # 2. Brand and Model Interaction
    if 'brand_model' not in df_engineered.columns and 'brand' in df_engineered.columns and 'model' in df_engineered.columns:
        df_engineered['brand_model'] = df_engineered['brand'].astype(str) + "_" + df_engineered['model'].astype(str)
        
    # 3. Usage Per Year
    if 'usage_per_year' not in df_engineered.columns and 'km_driven' in df_engineered.columns and 'car_age' in df_engineered.columns:
        df_engineered['usage_per_year'] = df_engineered['km_driven'] / (df_engineered['car_age'] + 1)
        
    # 4. Premium Brand Indicator
    if 'is_premium' not in df_engineered.columns and 'brand' in df_engineered.columns:
        df_engineered['is_premium'] = df_engineered['brand'].isin(PREMIUM_BRANDS).astype(int)
        
    # 5. Luxury Brand Indicator
    if 'is_luxury' not in df_engineered.columns and 'brand' in df_engineered.columns:
        df_engineered['is_luxury'] = df_engineered['brand'].isin(LUXURY_BRANDS).astype(int)
        
    # 6. Ownership Score
    if 'ownership_score' not in df_engineered.columns and 'owner_type' in df_engineered.columns:
        df_engineered['ownership_score'] = (
            df_engineered['owner_type']
            .astype(str)
            .str.lower()
            .map(OWNER_MAP)
            .fillna(1)
            .astype(int)
        )
        
    return df_engineered

def transform_features(df, preprocessor=None):
    """
    Transforms DataFrame features using a ColumnTransformer.
    If preprocessor is None, returns the DataFrame as-is.
    Otherwise, returns the encoded/scaled numpy array or DataFrame.
    """
    if preprocessor is None:
        return df
        
    # Standard column list expected by the preprocessor
    num_features = ['km_driven', 'car_age', 'usage_per_year', 'is_premium', 'is_luxury', 'ownership_score']
    cat_features = ['brand', 'model', 'transmission', 'owner_type', 'fuel_type', 'brand_model']
    required_features = num_features + cat_features
    
    # Ensure all required features are present
    df_filtered = df[required_features].copy()
    
    # Run ColumnTransformer transformation
    transformed = preprocessor.transform(df_filtered)
    
    return transformed
