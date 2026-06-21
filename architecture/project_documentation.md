# Car Price Prediction v2 - Technical Project Documentation

This document provides a comprehensive technical overview of the refactored **Car Price Prediction v2** architecture, listing pipelines, modeling steps, database schemas, and frontend aesthetics.

---

## 1. Directory Structure

The project has been restructured into a clean modular layout:

```
CarPricePredictorV2/
│
├── architecture/
│   ├── workflow_diagram.png          # Visual sequence of execution
│   └── project_documentation.md       # This documentation file
│
├── models/
│   ├── car_price_model_v2.joblib      # Full pipeline containing ColumnTransformer & CatBoost
│   └── preprocessing_pipeline.pkl     # Extracted preprocessor ColumnTransformer pickle
│
├── plots/
│   ├── correlation_heatmap.png        # Features correlation heatmap
│   ├── feature_importance.png         # CatBoost top 15 features plot
│   └── model_performance.png          # regression models comparison plot
│
├── src/
│   │
│   ├── frontend/
│   │   ├── templates/
│   │   │   ├── custom_styles.css      # Premium glassmorphic styles
│   │   │   └── ui_components.html     # HTML templates for metrics & prediction cards
│   │
│   └── backend/
│       ├── data_cleaning.py           # Dataset loading and standardization
│       ├── feature_engineering.py      # Calculations of interaction terms
│       ├── prediction.py              # Thread-safe model load and inference
│       └── database/
│           └── car_predictions.db     # SQLite logging database
│
├── main.py                            # End-to-end orchestration runner
├── app.py                             # Premium Streamlit web application
├── requirements.txt                   # Explicit python packages
└── run_app.bat                        # Batch execution entry point
```

---

## 2. Dataset Overview & Columns

The training data contains 14,993 records of cars from the Indian used car market. The raw features are:

| Raw Column | Standardized Name | Type | Description / Cleaning Operation |
|:---|:---|:---|:---|
| `Brand` | `brand` | Categorical | Manufacturer name. Standardized to text. |
| `model` | `model` | Categorical | Specific model name. Standardized to text. |
| `Year` | `year` | Numerical | Manufacturing Year. Used to compute car age. |
| `Age` | `car_age` | Numerical | Age in years. Derived as `current_year - year`. Outliers > 30 dropped. |
| `kmDriven` | `km_driven` | Numerical | Kilometers driven. Cleaned from currency/text strings like `"98,000 km"`. |
| `Transmission` | `transmission` | Categorical | `"Manual"` or `"Automatic"`. |
| `Owner` | `owner_type` | Categorical | Standardized to lowercase (`"first"`, `"second"`, etc.) for encoding. |
| `FuelType` | `fuel_type` | Categorical | `"Petrol"`, `"Diesel"`, `"CNG"`, `"LPG"`. |
| `AskPrice` | `selling_price` | Numerical | Target variable (Ask Price). Cleaned from characters like `"₹ 1,95,000"`. |
| `PostedDate` | - | Categorical | Omitted from model training features (redundant). |
| `AdditionInfo` | - | Text | Omitted from model training features (redundant). |

---

## 3. Data Cleaning Pipeline (`data_cleaning.py`)

- **Currency and Numeric Formatting:** Regular expressions extract digits and periods from `AskPrice` and `kmDriven` to transform text strings into floats.
- **Duplicate Removal:** Identical rows are dropped to prevent overfitting.
- **Missing Value Handling:** Missing target variables are dropped. Missing `km_driven` entries are imputed with the column median.
- **Outlier Filtering:** Car age is verified to be within `[0, 30)` years.

---

## 4. Feature Engineering (`feature_engineering.py`)

Advanced features were created to capture car interactions and depreciation:

1. **`car_age`:** `date.today().year - year` (depreciation factor).
2. **`brand_model`:** Concatenation `brand + "_" + model` to model specific brand depreciation trends.
3. **`usage_per_year`:** `km_driven / (car_age + 1)` (annual utilization index).
4. **`is_premium`:** Flag (1/0) indicating if the brand is in `['Honda', 'Toyota', 'Volkswagen', 'Skoda']`.
5. **`is_luxury`:** Flag (1/0) indicating if the brand is in `['Mercedes-Benz', 'BMW', 'Audi', 'Volvo', 'Jaguar', 'Land Rover']`.
6. **`ownership_score`:** Ordinal mapping of owner types: `{'first': 1, 'second': 2, 'third': 3, 'fourth & above': 4}`.

---

## 5. Machine Learning Model Architecture

### Pipeline Abstraction
The model is serialized as a composite scikit-learn `Pipeline` object containing:
1. **`preprocessor` (`ColumnTransformer`):**
   - **Numerical Pipeline:** `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
   - **Categorical Pipeline:** `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(handle_unknown='ignore')`.
2. **`regressor`:**
   - **CatBoostRegressor:** High-performance gradient boosting on decision trees optimized for categorical variables.

### Metrics Comparison
Models compared in reference notebook:

| Model Regressor | R² Score (R-squared) | MAE (Mean Absolute Error) | RMSE (Root Mean Sq. Error) |
|:---|:---:|:---:|:---:|
| **CatBoost (Selected)** | **0.8335** | **₹ 2,20,907.93** | **₹ 6,06,290.06** |
| XGBoost | 0.8222 | ₹ 2,16,680.79 | ₹ 6,26,481.20 |
| Random Forest | 0.8093 | ₹ 1,85,179.70 | ₹ 6,48,808.58 |
| Gradient Boosting | 0.7430 | ₹ 3,16,508.47 | ₹ 7,53,241.04 |

---

## 6. Database Schema (`database.py`)

Historical predictions are saved in a local SQLite database at `src/backend/database/car_predictions.db` under the `predictions` table:

```sql
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    year INTEGER NOT NULL,
    km_driven REAL NOT NULL,
    transmission TEXT NOT NULL,
    fuel_type TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    car_age INTEGER NOT NULL,
    predicted_price REAL NOT NULL,
    confidence REAL NOT NULL
);
```

---

## 7. Frontend Architecture & Aesthetics

The frontend uses Streamlit styled via:
- **`custom_styles.css`:** Implements CSS typography, background glassmorphism (`backdrop-filter`), glowing metrics boxes, custom form cards, hover animations.
- **`ui_components.html`:** Custom HTML templates formatted dynamically using Python to show prediction valuations, estimation ranges, and general database metrics.
- **Similar Listings Analytics:** Plots dynamic Seaborn charts using matching records from the training data to visualize similar vehicles' mileage vs price distributions.
