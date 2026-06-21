# Car Price Predictor v2 🚗

An industry-level, production-ready machine learning system to predict the selling price of used cars in the Indian automobile market.

This project refactors a raw analytical notebook into a professional architecture utilizing modular pipelines, SQLite prediction logging, and a premium glassmorphic Streamlit web application.

---

## 📂 Project Directory Structure

```
CarPricePredictorV2/
│
├── architecture/
│   ├── workflow_diagram.png          # Flowchart illustrating execution flow
│   └── project_documentation.md       # Full technical details of the pipeline
│
├── models/
│   ├── car_price_model_v2.joblib      # CatBoost inference pipeline
│   └── preprocessing_pipeline.pkl     # Extracted ColumnTransformer object
│
├── plots/
│   ├── correlation_heatmap.png        # Features correlation heatmap
│   ├── feature_importance.png         # Model feature importances plot
│   └── model_performance.png          # Comparison chart of regressor models
│
├── src/
│   ├── frontend/
│   │   └── templates/
│   │       ├── custom_styles.css      # Custom UI glassmorphic styles
│   │       └── ui_components.html     # Reusable HTML metric card templates
│   │
│   └── backend/
│       ├── data_cleaning.py           # Dataset loading & cleaning pipelines
│       ├── feature_engineering.py      # Calculations of interaction terms
│       ├── prediction.py              # Load model and run prediction
│       └── database.py                # SQLite predictions logging utility
│
├── main.py                            # End-to-end orchestration runner
├── app.py                             # Premium Streamlit UI application
├── requirements.txt                   # Project package dependencies
└── run_app.bat                        # Double-click application launcher (Windows)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Windows OS (for running the batch script)

### Setup Steps
1. **Initialize Virtual Environment:**
   If a virtual environment is not already present, create one:
   ```bash
   python -m venv .venv
   ```

2. **Install Dependencies:**
   Ensure dependencies from `requirements.txt` are fully installed:
   ```bash
   .venv\Scripts\pip.exe install -r requirements.txt
   ```

3. **Verify and Initialize Pipeline:**
   Run the master orchestration script to clean data, log tests to database, generate visualization plots, and check model inferences:
   ```bash
   .venv\Scripts\python.exe main.py
   ```

---

## 🚀 Execution Instructions

### Run the Web Application
To run the Streamlit web application dashboard:

- **Windows:** Double-click the `run_app.bat` file in the project root.
- **Terminal:** Run the following command:
  ```bash
  .venv\Scripts\streamlit run app.py
  ```

---

## 📊 Pipeline Workflow

The project executes following the exact sequence diagram below (saved under [workflow_diagram.png](file:///c:/Users/acer/Desktop/CarPricePredictorV2/architecture/workflow_diagram.png)):

```
run_app.bat
    ↓
app.py (Loads preprocessor, model, templates)
    ↓
data_cleaning.py (Cleans & standardizes inputs)
    ↓
feature_engineering.py (Applies interaction transformations)
    ↓
models/car_price_model_v2.joblib (Executes CatBoost regressor)
    ↓
database.py (Saves inputs and outputs into SQLite predictions.db)
    ↓
prediction.py (Returns predicted price & depreciation metrics)
    ↓
Premium UI Dashboard (Displays stats, similar cars scatter plots, & logging logs)
```

---

## 📈 Model Performance Highlights

The system leverages a **CatBoost Regressor** trained on 14,993 used car listings, outperforming other regression models compared during development:

- **R² Score:** 0.8335 (83.35% of price variance explained)
- **Mean Absolute Error (MAE):** ₹220,907.93
- **Root Mean Squared Error (RMSE):** ₹606,290.06
