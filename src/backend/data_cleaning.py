import pandas as pd
import numpy as np
import re

def load_dataset(filepath):
    """
    Loads the dataset from a CSV file.
    """
    return pd.read_csv(filepath)

def clean_currency(value):
    """
    Removes currency symbols, commas, and converts value to float.
    """
    if isinstance(value, str):
        value = re.sub(r'[^0-9.]', '', value)
        return float(value) if value else np.nan
    return float(value) if value is not None else np.nan

def clean_km(value):
    """
    Removes 'km', commas, and converts value to float.
    """
    if isinstance(value, str):
        value = value.lower().replace('km', '')
        value = re.sub(r'[^0-9.]', '', value)
        return float(value) if value else np.nan
    return float(value) if value is not None else np.nan

def clean_dataset(df):
    """
    Cleans dataset columns: standardizes column names, cleans currency and mileage formats,
    removes duplicates, and fills missing values.
    """
    # Standardize column mapping to lowercase/snake_case
    rename_map = {
        'Brand': 'brand',
        'Year': 'year',
        'kmDriven': 'km_driven',
        'Transmission': 'transmission',
        'Owner': 'owner_type',
        'FuelType': 'fuel_type',
        'AskPrice': 'selling_price'
    }
    
    # Copy DataFrame to avoid modifying in-place unexpectedly
    df_cleaned = df.copy()
    
    # Rename columns if they exist in the rename map
    df_cleaned = df_cleaned.rename(columns={k: v for k, v in rename_map.items() if k in df_cleaned.columns})
    
    # Ensure correct categories exist
    if 'transmission' in df_cleaned.columns:
        df_cleaned['transmission'] = df_cleaned['transmission'].astype(str)
    if 'owner_type' in df_cleaned.columns:
        df_cleaned['owner_type'] = df_cleaned['owner_type'].astype(str)
    if 'fuel_type' in df_cleaned.columns:
        df_cleaned['fuel_type'] = df_cleaned['fuel_type'].astype(str)
        
    # Clean numeric columns
    if 'km_driven' in df_cleaned.columns:
        df_cleaned['km_driven'] = df_cleaned['km_driven'].apply(clean_km)
        # Handle missing values for km_driven using median
        median_km = df_cleaned['km_driven'].median()
        if pd.isna(median_km):
            median_km = 50000.0  # Safe default if all is NaN
        df_cleaned['km_driven'] = df_cleaned['km_driven'].fillna(median_km)
        
    if 'selling_price' in df_cleaned.columns:
        df_cleaned['selling_price'] = df_cleaned['selling_price'].apply(clean_currency)
        df_cleaned = df_cleaned.dropna(subset=['selling_price'])
        
    # Drop duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    return df_cleaned

def remove_outliers(df):
    """
    Removes car age outliers. Fits to dataset that has 'car_age' or 'year' column.
    """
    df_filtered = df.copy()
    
    if 'car_age' not in df_filtered.columns and 'year' in df_filtered.columns:
        from datetime import date
        current_year = date.today().year
        df_filtered['car_age'] = current_year - df_filtered['year']
        
    if 'car_age' in df_filtered.columns:
        df_filtered = df_filtered[(df_filtered['car_age'] >= 0) & (df_filtered['car_age'] < 30)]
        
    return df_filtered

def prepare_features(df):
    """
    Validates features and ensures standard datatypes before feeding into model pipeline.
    """
    df_prepared = df.copy()
    
    # Standardize column types
    if 'km_driven' in df_prepared.columns:
        df_prepared['km_driven'] = pd.to_numeric(df_prepared['km_driven'], errors='coerce')
    if 'selling_price' in df_prepared.columns:
        df_prepared['selling_price'] = pd.to_numeric(df_prepared['selling_price'], errors='coerce')
        df_prepared = df_prepared.dropna(subset=['selling_price'])
        
    return df_prepared
