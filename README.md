# Advance-machine-Learning-On-AutoCar-Tarder-Data-set

End-to-end machine learning pipeline to predict used/new car prices from the **adverts.csv** dataset.  
The project covers **data cleaning**, **feature engineering**, **feature selection**, **dimensionality reduction (PCA)**, **multiple models** (Linear/Ridge, Random Forest, Gradient Boosting), an **ensemble**, and **model explainability** with **SHAP** and **Partial Dependence Plots (PDPs)**.

> ⚠️ Note: Metrics in this README are computed on **Min-Max scaled targets** (price scaled to 0–1). They’re comparable across experiments here, but they’re not in £. See “Interpreting Metrics” for how to get values back in £.

---

## ✨ Highlights

- Cleans 400k+ rows, imputes missing values, and removes outliers
- Encodes categorical features, engineers domain features (`miles_per_year_interaction`, `car_age`), and builds polynomial interactions
- Compares models:
  - **Linear Regression / Ridge**
  - **Random Forest** (+GridSearch)
  - **Gradient Boosting** (+GridSearch)
  - **VotingRegressor Ensemble**
- Cross-validated evaluation (**R², RMSE, MAE, MSE**)
- **Explainability** via **SHAP** (global & local) and **PDPs**
- Reproducible split and sampling (`random_state=42`)

---

## ✅ Requirements

Tested with Python **3.10** and these libraries:

```
numpy==1.25.2
pandas==2.0.3
scikit-learn==1.2.2
matplotlib==3.7.3
seaborn==0.13.2
shap==0.45.1
scipy==1.11.4
tqdm==4.66.4
numba==0.58.1
llvmlite==0.41.1
packaging==24.0
```

Create a `requirements.txt` using the block above.

---

## ⚙️ Setup

### Option A) Virtualenv (Windows/macOS/Linux)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Option B) Google Colab

If running in Colab, install anything missing (often SHAP is already there):

```python
!pip install shap
```

Place `adverts.csv` in your working directory or mount Drive accordingly.

---

## 🚀 How to Run

### If you have a Python script (e.g., `main.py`)

```bash
python main.py
```

### If you have a Jupyter notebook

Open the notebook and run all cells in order.

**Input**: `adverts.csv`  
**Outputs**:
- Cleaned and down-sampled data → `sampled_data.csv`
- Plots: boxplots, CV curves, True vs Predicted scatter, SHAP (waterfall, beeswarm), PDPs

---

## 🧹 Data Processing

1. **Missing values**
   - `mileage`:  
     - `vehicle_condition == "NEW"` → set to `0`  
     - `vehicle_condition == "USED"` → impute with mean mileage of USED cars
   - `year_of_registration`: `NEW` cars set to **2024** (assumed “current” reg)
   - `reg_code`: fill `0` (unused later, dropped)
   - Drop remaining rows with missing values after above imputations

2. **Outliers**
   - Z-score filtering (|z| < **2**) on `mileage` and `price`  
     *(comment said 3; code uses 2)*

3. **Down-sampling**
   - Keep **10%** (`data.sample(frac=0.1, random_state=42)`) → `sampled_data.csv`

4. **Drop unused columns**
   - `public_reference`, `reg_code`, `crossover_car_and_van`

5. **Categorical encoding**
   - `LabelEncoder` on: `standard_make`, `standard_model`, `vehicle_condition`, `standard_colour`, `body_type`, `fuel_type`

---

## 🏗️ Feature Engineering

- `miles_per_year_interaction = mileage / (2024 - year_of_registration)`  
  (filled `0` and inf handled)
- `car_age = 2024 - year_of_registration`
- **Polynomial interactions** (degree=2) for: `['mileage', 'year_of_registration']`  
  → `mileage^2`, `mileage year_of_registration`, `year_of_registration^2`

---

## 🔎 Feature Selection & Dimensionality Reduction

- **Scaling**: `MinMaxScaler` on all modeling features (including engineered ones)
- **Train/Test Split**: 80/20, `random_state=42`
- **SelectKBest (f_regression)**: selects top **5** features:
  - `mileage`, `vehicle_condition`, `year_of_registration`, `miles_per_year_interaction`, `car_age`
- **PCA**: `n_components=2` (added as optional analysis; not used in X train for models by default)

---

## 🤖 Models

1. **Linear Regression**
2. **RidgeCV** (alphas = `[0.01, 0.1, 1, 10]`, cv=5)
3. **Random Forest Regressor**  
   - Baseline: `n_estimators=100, random_state=42`  
   - GridSearch: `n_estimators=[5,10,20]`, `max_depth=[None,10,20]`, `min_samples_split=[2,5,10]`, `min_samples_leaf=[1,2,4]`
4. **Gradient Boosting Regressor**  
   - Baseline: `n_estimators=100, random_state=42`  
   - GridSearch: `n_estimators=[5,10,20]`, `learning_rate=[0.01,0.1,0.2]`, `max_depth=[3,5,7]`, `min_samples_split=[2,5,10]`, `min_samples_leaf=[1,2,4]`
5. **Ensemble**  
   - `VotingRegressor` over (Linear, RandomForest, GradientBoosting)

---

## 📊 Evaluation

### Single Train/Test Split (example snapshot)

- **Linear Regression**  
  R² ≈ **0.4175**, RMSE ≈ **0.1299**, MAE ≈ **0.0913**
- **Ridge**  
  R² ≈ **0.4155**, RMSE ≈ **0.1301**, MAE ≈ **0.0915**
- **Random Forest**  
  R² ≈ **0.8982**, RMSE ≈ **0.0543**, MAE ≈ **0.0320**
- **Random Forest (GridSearch best)**  
  `{'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 20}`  
  R² ≈ **0.8929**, RMSE ≈ **0.0557**, MAE ≈ **0.0324**
- **Gradient Boosting**  
  R² ≈ **0.7599**, RMSE ≈ **0.0834**, MAE ≈ **0.0516**
- **Gradient Boosting (GridSearch best)**  
  `{'learning_rate': 0.2, 'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 20}`  
  R² ≈ **0.8607**, RMSE ≈ **0.0635**, MAE ≈ **0.0394**
- **Ensemble (VotingRegressor)**  
  R² ≈ **0.7918**, RMSE ≈ **0.0776**, MAE ≈ **0.0499**

### 5-Fold Cross-Validation (example snapshot)

- **Linear Regression**  
  R² ≈ **0.3322**, RMSE ≈ **0.1387**, MAE ≈ **0.0927**
- **Random Forest**  
  R² ≈ **0.8982**, RMSE ≈ **0.0545**, MAE ≈ **0.0323**
- **Gradient Boosting**  
  R² ≈ **0.7614**, RMSE ≈ **0.0834**, MAE ≈ **0.0524**
- **Ensemble**  
  R² ≈ **0.7808**, RMSE ≈ **0.0799**, MAE ≈ **0.0512**

> ✅ **Best overall** in this setup: **Random Forest** (both split and CV).

---

## 🧠 Explainability

### SHAP

- **Linear model**: `LinearExplainer` (global coefficients + local waterfalls)
- **Tree models (RF/GB)**: `TreeExplainer` (beeswarm for global, waterfall for local)

Example top importances (RF, indicative):

```
year_of_registration, car_age, standard_model, standard_make, body_type, ...
```

### Partial Dependence Plots (PDPs)

- PDPs for `mileage`, `year_of_registration`, `car_age`, `fuel_type`, and `miles_per_year_interaction`
- Run via `sklearn.inspection.PartialDependenceDisplay.from_estimator(...)`

---

## 📈 Interpreting Metrics & Prices

All metrics were computed on **scaled price** (0–1). To convert predictions back to £:

```python
from sklearn.preprocessing import MinMaxScaler

# Fit scaler on the original price column BEFORE scaling
price_scaler = MinMaxScaler()
price_scaler.fit(data[['price']])  # data before scaling

# After model prediction (y_pred_scaled), invert:
y_pred_pounds = price_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
```

---

## 🧪 Reproducibility

- Random seeds set with `random_state=42` for sampling, train/test split, and certain models
- Results may vary slightly by OS/BLAS/parallelism

---

## ⚠️ Notes & Assumptions

- `year_of_registration` for `NEW` vehicles set to **2024** as a practical proxy
- Outlier filter uses **|z| < 2** (aggressive); adjust if you prefer a wider cut (e.g., 3)
- `LabelEncoder` is used for categorical features; for linear models, one-hot encoding could perform better and avoid ordinal assumptions
- Metrics are reported on **scaled target**; consider inverse-scaling for business reporting

---

## 📷 Generated Visuals

- Boxplots (pre/post outlier removal) for `mileage` and `price`
- Cross-validation line charts (R², RMSE, MAE, MSE per fold)
- True vs Predicted scatter plots
- SHAP:
  - Waterfall plots (local explanations)
  - Beeswarm plots (global feature influence)
- PDPs for key features

> Some SHAP plots on large samples can be slow. In the code, a smaller sample (e.g., 100 rows) is used when needed.

---

## 🧩 Next Steps / Ideas

- Switch to **One-Hot Encoding** for categorical features (esp. for linear/Ridge)
- Calibrate models and **inverse-scale** predictions to £ for business use
- Try **XGBoost/LightGBM/CatBoost** for potentially stronger performance
- Add **model persistence** (joblib), and a lightweight **FastAPI** for inference
- Add **MLflow** tracking for experiments

---

## 📄 License

MIT (feel free to reuse with attribution)

---

## 🙌 Acknowledgments

- [scikit-learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/)

---

## 👤 Author

**Vinay Mandora**  
Manchester, UK
