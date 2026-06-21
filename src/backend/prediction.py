import joblib
import pandas as pd
import numpy as np

def load_model(model_path):
    """
    Loads the trained scikit-learn pipeline model from the joblib file.
    """
    return joblib.load(model_path)

def predict_price(model, input_df):
    """
    Takes a loaded pipeline model and a pandas DataFrame containing raw inputs,
    verifies feature name order and columns, runs prediction, and returns the predicted selling price.
    """
    # Standard column list expected by the pipeline model
    expected_cols = [
        'brand', 'model', 'km_driven', 'transmission', 'owner_type', 'fuel_type',
        'car_age', 'brand_model', 'usage_per_year', 'is_premium', 'is_luxury',
        'ownership_score'
    ]
    
    # Ensure columns exist and are ordered exactly as the model was trained
    input_df_reordered = input_df.copy()
    
    # Check for missing columns and populate with default values if necessary
    for col in expected_cols:
        if col not in input_df_reordered.columns:
            if col in ['km_driven', 'car_age', 'usage_per_year']:
                input_df_reordered[col] = 0.0
            elif col in ['is_premium', 'is_luxury', 'ownership_score']:
                input_df_reordered[col] = 0
            else:
                input_df_reordered[col] = "unknown"
                
    # Select and reorder columns
    input_df_reordered = input_df_reordered[expected_cols]
    
    # Clean data types to ensure alignment
    input_df_reordered['km_driven'] = input_df_reordered['km_driven'].astype(float)
    input_df_reordered['car_age'] = input_df_reordered['car_age'].astype(float)
    input_df_reordered['usage_per_year'] = input_df_reordered['usage_per_year'].astype(float)
    input_df_reordered['is_premium'] = input_df_reordered['is_premium'].astype(int)
    input_df_reordered['is_luxury'] = input_df_reordered['is_luxury'].astype(int)
    input_df_reordered['ownership_score'] = input_df_reordered['ownership_score'].astype(float)
    
    # Run prediction through the full pipeline
    prediction = model.predict(input_df_reordered)
    
    # Extract prediction value
    if isinstance(prediction, (np.ndarray, list, pd.Series)):
        price = float(prediction[0])
    else:
        price = float(prediction)
        
    return price
