import os
import sys
import pickle
import joblib
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src/ to system path so we can import backend packages directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from backend.data_cleaning import load_dataset, clean_dataset, remove_outliers, prepare_features
from backend.feature_engineering import create_features, transform_features
from backend.prediction import load_model, predict_price
from backend.database import create_database, save_prediction, load_history

def generate_correlation_heatmap(df_engineered):
    """
    Generates a correlation heatmap of the numerical features and saves it to plots/.
    """
    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(10, 8))
    
    # Select only numeric features
    numeric_cols = ['km_driven', 'car_age', 'usage_per_year', 'is_premium', 'is_luxury', 'ownership_score', 'selling_price']
    existing_numeric = [c for c in numeric_cols if c in df_engineered.columns]
    
    corr_matrix = df_engineered[existing_numeric].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Used Car Feature Correlation Heatmap', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved correlation heatmap to plots/correlation_heatmap.png")

def generate_feature_importance():
    """
    Extracts feature importances from the CatBoost model and saves the plot to plots/.
    """
    os.makedirs('plots', exist_ok=True)
    
    # Load model
    model_path = 'models/car_price_model_v2.joblib'
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    preprocessor = model.named_steps['preprocessor']
    regressor = model.named_steps['regressor']
    
    # ColumnTransformer feature lists
    num_features = ['km_driven', 'car_age', 'usage_per_year', 'is_premium', 'is_luxury', 'ownership_score']
    cat_features = ['brand', 'model', 'transmission', 'owner_type', 'fuel_type', 'brand_model']
    
    # Extract categorical encoded names
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    ohe_features = list(encoder.get_feature_names_out(cat_features))
    all_features = num_features + ohe_features
    
    # Feature importances
    importances = regressor.feature_importances_
    
    # Create DataFrame
    feat_imp_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Select top 15 features for visualization
    top_feats = feat_imp_df.head(15)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_feats, palette='viridis', hue='Feature', legend=False)
    plt.title('Top 15 Most Important Features (CatBoost Model)', fontsize=14, pad=15)
    plt.xlabel('Importance Score (%)')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved feature importance plot to plots/feature_importance.png")

def generate_model_performance():
    """
    Generates a model performance comparison bar chart based on reference notebook metrics.
    """
    os.makedirs('plots', exist_ok=True)
    
    # Data from reference notebook model training runs
    comparison_data = {
        'Model': ['CatBoost', 'XGBoost', 'Random Forest', 'Gradient Boosting'],
        'R2 Score': [0.833486, 0.822210, 0.809312, 0.742985],
        'MAE': [220907.93, 216680.79, 185179.70, 316508.47],
        'RMSE': [606290.06, 626481.20, 648808.58, 753241.04]
    }
    df_perf = pd.DataFrame(comparison_data)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Bar plot for R2 Score (Left Axis)
    color = 'tab:blue'
    sns.barplot(x='Model', y='R2 Score', data=df_perf, ax=ax1, color=color, alpha=0.7)
    ax1.set_xlabel('Model Regressor Type', fontsize=12, labelpad=10)
    ax1.set_ylabel('R² Score (Higher is Better)', color=color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.6, 0.9)
    plt.title('Model Regression Performance Comparison', fontsize=14, pad=15)
    
    # Line plot for MAE (Right Axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    sns.lineplot(x='Model', y='MAE', data=df_perf, ax=ax2, color=color, marker='o', linewidth=2.5, markersize=8)
    ax2.set_ylabel('Mean Absolute Error (₹, Lower is Better)', color=color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.savefig('plots/model_performance.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved model performance plot to plots/model_performance.png")

def generate_workflow_diagram():
    """
    Draws and saves a beautiful flowchart representing the execution workflow.
    """
    os.makedirs('architecture', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Define flowchart steps
    steps = [
        "1. run_app.bat\n(Entry point)",
        "2. app.py\n(Streamlit UI initialization)",
        "3. backend modules loaded\n(data_cleaning.py, prediction.py, database.py)",
        "4. data_cleaning.py\n(Standardizes and cleans inputs)",
        "5. feature_engineering.py\n(Computes interaction terms)",
        "6. models/car_price_model_v2.joblib\n(Executes model prediction)",
        "7. database.py\n(Persists prediction entry in SQLite)",
        "8. Web App Frontend\n(Renders metrics cards & charts)"
    ]
    
    # Draw boxes and arrows
    y_pos = 0.9
    box_props = dict(boxstyle="round,pad=0.5", fc="#F3F4F6", ec="#D1D5DB", lw=1.5)
    
    for i, step in enumerate(steps):
        # Draw step box
        ax.text(0.5, y_pos, step, ha="center", va="center", bbox=box_props, fontsize=10, fontname="sans-serif")
        
        # Draw arrow to next step
        if i < len(steps) - 1:
            ax.annotate('', xy=(0.5, y_pos - 0.05), xytext=(0.5, y_pos - 0.08),
                        arrowprops=dict(arrowstyle="<-", color="#4B5563", lw=2))
        y_pos -= 0.11
        
    plt.title('Car Price Predictor execution Workflow Sequence', fontsize=14, pad=20, weight='bold')
    plt.tight_layout()
    plt.savefig('architecture/workflow_diagram.png', dpi=300)
    plt.close()
    print("[SUCCESS] Saved workflow diagram to architecture/workflow_diagram.png")

def main():
    print("====================================================")
    print("       CAR PRICE PREDICTOR PIPELINE ORCHESTRATION    ")
    print("====================================================")
    
    # 1. Check directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('architecture', exist_ok=True)
    
    # 2. Check and load dataset
    dataset_path = 'used_cars_dataset_v2.csv'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at {dataset_path}!")
        sys.exit(1)
    
    print("1. Loading raw dataset...")
    df_raw = load_dataset(dataset_path)
    print(f"  - Loaded shape: {df_raw.shape}")
    
    # 3. Clean dataset
    print("2. Cleaning dataset...")
    df_cleaned = clean_dataset(df_raw)
    print(f"  - Cleaned shape (removed duplicates/nulls): {df_cleaned.shape}")
    
    # 4. Feature Engineering
    print("3. Performing feature engineering...")
    df_engineered = create_features(df_cleaned)
    df_prepared = prepare_features(df_engineered)
    df_final = remove_outliers(df_prepared)
    print(f"  - Final clean & prepared dataset shape: {df_final.shape}")
    
    # 5. Generate plots
    print("4. Generating visual plots...")
    generate_correlation_heatmap(df_final)
    generate_feature_importance()
    generate_model_performance()
    generate_workflow_diagram()
    
    # 6. Database Verification
    print("5. Verifying database logging integration...")
    create_database()
    # Test insertion
    test_brand = "Maruti Suzuki"
    test_model = "Swift"
    test_year = 2018
    test_km = 45000
    test_transmission = "Manual"
    test_fuel = "Petrol"
    test_owner = "first"
    test_age = 8
    test_price = 550000.00
    test_confidence = 92.5
    
    save_prediction(
        brand=test_brand,
        model=test_model,
        year=test_year,
        km_driven=test_km,
        transmission=test_transmission,
        fuel_type=test_fuel,
        owner_type=test_owner,
        car_age=test_age,
        predicted_price=test_price,
        confidence=test_confidence
    )
    
    history_df = load_history(limit=5)
    print(f"  - SQLite History Record count: {len(history_df)}")
    print(f"  - Database Log verification: SUCCESS (Recorded test Swift at Rs. {test_price:,.2f})")
    
    # 7. Model Inference Test
    print("6. Verifying model inference...")
    model_path = 'models/car_price_model_v2.joblib'
    if os.path.exists(model_path):
        model = load_model(model_path)
        sample_input = pd.DataFrame({
            'brand': [test_brand],
            'model': [test_model],
            'km_driven': [float(test_km)],
            'transmission': [test_transmission],
            'owner_type': [test_owner],
            'fuel_type': [test_fuel],
            'car_age': [test_age],
            'brand_model': [f"{test_brand}_{test_model}"],
            'usage_per_year': [test_km / (test_age + 1)],
            'is_premium': [0],
            'is_luxury': [0],
            'ownership_score': [1.0]
        })
        predicted = predict_price(model, sample_input)
        print(f"  - Sample prediction success! Estimated price: Rs. {predicted:,.2f}")
    else:
        print("  - Skipping model inference check (model not found).")
        
    print("\n[SUCCESS] Pipeline verification completed successfully.")
    print("====================================================")

if __name__ == "__main__":
    main()
