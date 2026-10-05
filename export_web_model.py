import os
import json
import re
import joblib
import pandas as pd
import numpy as np

def export():
    os.makedirs('docs', exist_ok=True)
    
    # 1. Load trained pipeline
    model_path = 'models/car_price_model_v2.joblib'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found!")
    
    pipeline = joblib.load(model_path)
    preprocessor = pipeline.named_steps['preprocessor']
    regressor = pipeline.named_steps['regressor']
    
    # 2. Extract CatBoost trees to JSON
    temp_cb_json = 'docs/temp_catboost.json'
    regressor.save_model(temp_cb_json, format='json')
    with open(temp_cb_json, 'r', encoding='utf-8') as f:
        cb_raw = json.load(f)
    os.remove(temp_cb_json)
    
    compact_trees = []
    for tree in cb_raw['oblivious_trees']:
        splits = [[s['float_feature_index'], round(float(s['border']), 5)] for s in tree['splits']]
        leaves = [round(float(v), 2) for v in tree['leaf_values']]
        compact_trees.append({'s': splits, 'l': leaves})
        
    bias = round(float(cb_raw['scale_and_bias'][1][0]), 2)
    
    # 3. Scaler and Imputer parameters
    scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
    means = [round(float(x), 5) for x in scaler.mean_.tolist()]
    scales = [round(float(x), 5) for x in scaler.scale_.tolist()]
    
    # 4. Categorical encoder mapping
    encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
    cat_cols = ['brand', 'model', 'transmission', 'owner_type', 'fuel_type', 'brand_model']
    cat_map = {}
    offset = 6  # 6 numerical features come first
    for i, col in enumerate(cat_cols):
        cats = encoder.categories_[i]
        col_dict = {}
        for cat_val in cats:
            col_dict[str(cat_val)] = offset
            offset += 1
        cat_map[col] = col_dict
        
    # 5. Extract available brands and models from dataset
    dataset_path = 'used_cars_dataset_v2.csv'
    df = pd.read_csv(dataset_path)
    
    brands_models = {}
    for brand in sorted(df['Brand'].dropna().unique()):
        models = sorted(df[df['Brand'] == brand]['model'].dropna().unique().tolist())
        brands_models[str(brand)] = [str(m) for m in models]
        
    # Calculate dataset stats
    def clean_curr(v):
        s = re.sub(r'[^0-9.]', '', str(v))
        return float(s) if s else None
        
    prices = df['AskPrice'].apply(clean_curr).dropna()
    avg_price = round(float(prices.mean()), 2)
    
    model_payload = {
        'bias': bias,
        'means': means,
        'scales': scales,
        'cat_map': cat_map,
        'trees': compact_trees,
        'brands_models': brands_models,
        'total_cars': len(df),
        'avg_price': avg_price
    }
    
    with open('docs/model_data.json', 'w', encoding='utf-8') as f:
        json.dump(model_payload, f)
    print(f"[SUCCESS] Exported model_data.json ({os.path.getsize('docs/model_data.json') / 1024:.1f} KB)")
    
    # 6. Export cars points for similar car scatter chart
    def clean_km(v):
        s = re.sub(r'[^0-9.]', '', str(v).lower().replace('km', ''))
        return int(float(s)) if s else None
        
    clean_cars = []
    for _, r in df.iterrows():
        p = clean_curr(r['AskPrice'])
        k = clean_km(r['kmDriven'])
        if p and k and p > 10000 and k < 500000:
            clean_cars.append([
                str(r['Brand']),
                str(r['model']),
                int(r['Year']),
                int(k),
                int(p)
            ])
            
    with open('docs/cars_data.json', 'w', encoding='utf-8') as f:
        json.dump(clean_cars, f)
    print(f"[SUCCESS] Exported cars_data.json ({os.path.getsize('docs/cars_data.json') / 1024:.1f} KB, {len(clean_cars)} records)")

if __name__ == '__main__':
    export()
