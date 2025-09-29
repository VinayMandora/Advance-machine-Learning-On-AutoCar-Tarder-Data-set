# Advance-machine-Learning-On-AutoCar-Tarder-Data-set

End-to-end machine-learning workflow to predict **used car prices** from structured advert data.  
The project covers **data cleaning**, **feature engineering** (incl. polynomial features), **modeling** (Linear/Ridge, Random Forest, Gradient Boosting, Voting Ensemble), **model selection via CV**, and **explainability** using **SHAP** and **Partial Dependence Plots (PDP)**.

> **Dataset**: `adverts.csv` (402,005 rows × 12 columns).  
> Target: `price` (numeric).

---

## ✨ Highlights

- Robust preprocessing for missing/erroneous values
- Outlier trimming (Z-score)
- Categorical encoding (LabelEncoder)
- Feature engineering: age, miles-per-year, polynomial interactions
- Train/test split + 5-fold cross-validation
- Multiple models with metrics (R², RMSE, MAE, MSE)
- Explainability: **SHAP** (global & local) and **PDP** per model
- Reproducible scripts/notebook cells

---

## 🧮 Features & Columns (incoming)

- `mileage` (float)
- `reg_code` (object) → filled `0` then dropped later
- `standard_colour` (object)
- `standard_make` (object)
- `standard_model` (object)
- `vehicle_condition` (object: NEW/USED)
- `year_of_registration` (float)
- `price` (int, target)
- `body_type` (object)
- `crossover_car_and_van` (bool) → dropped
- `fuel_type` (object)
- `public_reference` (int) → dropped

---

## 🧹 Data Cleaning

- **Mileage**:
  - If `vehicle_condition == 'NEW'` and `mileage` is null → set to `0`
  - If `vehicle_condition == 'USED'` and `mileage` is null → set to **mean mileage of USED** cars
- **Year of registration**:
  - If `vehicle_condition == 'NEW'` and null → set to `2024`
- **Categoricals**:
  - `reg_code` null → `0` (later dropped)
- **Drop remaining nulls** (rows)
- **Outliers**: remove with simple **Z-score** filter (`|z| < 2`) on `mileage`, `price`
- **Sampling**: take a **10% random sample** for faster experimentation → `sampled_data.csv`

> ⚠️ Notes:
> - Z-score with threshold 2 is aggressive; tune for your use case.
> - Check distribution shift if you sample. Keep random seed fixed for reproducibility.

---

## 🔧 Feature Engineering

- **Label encoding** for: `standard_make`, `standard_model`, `vehicle_condition`, `standard_colour`, `body_type`, `fuel_type`
- **Derived features**:
  - `car_age = 2024 - year_of_registration`
  - `miles_per_year_interaction = mileage / (2024 - year_of_registration)` (fill/inf → 0)
- **Polynomial features** (`degree=2`) for `['mileage','year_of_registration']`:
  - `mileage^2`, `year_of_registration^2`, `mileage × year_of_registration`

---

## 📉 Feature Scaling

- **MinMaxScaler** applied to the *feature matrix*.  
  ✅ Recommended: **fit the scaler on the training set only**, then transform the test set.  
  ❗ In early experiments it was fit on the full dataset (including `price`) which is **data leakage**. See “Caveats” below for the fix.

---

## 🔀 Train/Test Split

- `train_test_split(test_size=0.2, random_state=42)`  
- **K-Best** univariate selection (`k=5` via `f_regression`) used before optional PCA.
- Optional **PCA (2 components)** for visualization/experiments (not required by tree models).

**Selected top features in one run:**  
`['mileage', 'vehicle_condition', 'year_of_registration', 'miles_per_year_interaction', 'car_age']`

---

## 🤖 Models

1. **Linear Regression**
2. **RidgeCV** (α in `[0.01, 0.1, 1, 10]`)
3. **RandomForestRegressor**  
   - Manual run: `n_estimators=100`  
   - GridSearchCV search space: `n_estimators=[5,10,20]`, `max_depth=[None,10,20]`, `min_samples_split=[2,5,10]`, `min_samples_leaf=[1,2,4]`
4. **GradientBoostingRegressor**  
   - GridSearchCV search space: `n_estimators=[5,10,20]`, `learning_rate=[0.01,0.1,0.2]`, `max_depth=[3,5,7]`, `min_samples_split=[2,5,10]`, `min_samples_leaf=[1,2,4]`
5. **VotingRegressor** (Linear + RF + GBoost)

> All metrics below are on **scaled target** (0–1). For business interpretation, train on original target units or invert scaling.

---

## 📊 Results (example run)

### Hold-out test (single split)
- **Linear Regression**: R² **0.417**, RMSE **0.130**, MAE **0.091**
- **Ridge**: R² **0.416**, RMSE **0.130**, MAE **0.092**
- **Random Forest**: R² **0.898**, RMSE **0.054**, MAE **0.032**
- **RF (GridSearchCV best)**: R² **0.893**, RMSE **0.056**, MAE **0.032**
- **Gradient Boosting**: R² **0.760**, RMSE **0.083**, MAE **0.052**
- **GB (GridSearchCV best)**: R² **0.861**, RMSE **0.063**, MAE **0.039**
- **Voting Ensemble**: R² **0.792**, RMSE **0.078**, MAE **0.050**

### 5-fold Cross-Validation (mean scores)
- **Linear**: R² **0.332**, RMSE **0.139**, MAE **0.093**
- **Random Forest**: R² **0.898**, RMSE **0.054**, MAE **0.032**
- **Gradient Boosting**: R² **0.761**, RMSE **0.083**, MAE **0.052**
- **Ensemble**: R² **0.781**, RMSE **0.080**, MAE **0.051**

> **Winner:** Random Forest by a margin (both hold-out and CV).

---

## 🧠 Explainability

### SHAP
- **Linear model**: `shap.LinearExplainer` for global coefficients + local waterfalls.
- **Tree models (RF, GB)**: `shap.TreeExplainer` for local waterfalls + global beeswarm plots.
- Key global drivers (typical run):
  - `year_of_registration` / `car_age`
  - `standard_make` / `standard_model`
  - `body_type`
  - `mileage`
  - `fuel_type`

### Partial Dependence Plots (PDP)
- Single & multi-feature PDPs for: `mileage`, `year_of_registration`, `car_age`, `fuel_type`, `miles_per_year_interaction`.
- Implemented via `sklearn.inspection.PartialDependenceDisplay`.

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
