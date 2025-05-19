import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Load dataset
df = pd.read_csv("Ecommerce_data.csv", encoding='latin-1')

# Drop rows with missing target values
df = df.dropna(subset=['sales_per_order', 'profit_per_order'])

# Label encode categorical features
label_cols = ['category_name', 'customer_region', 'shipping_type']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features for both models
features = ['category_name', 'customer_region', 'shipping_type', 'order_quantity', 'days_for_shipment_scheduled']

# ---------------------
# Model 1: Sales Prediction (Regression)
# ---------------------
X1 = df[features]
y1 = df['sales_per_order']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Use RandomForestRegressor with hyperparameter tuning
param_grid_regressor = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

reg_model = RandomForestRegressor(random_state=42)
reg_search = RandomizedSearchCV(reg_model, param_grid_regressor, n_iter=10, cv=3, random_state=42, n_jobs=-1)
reg_search.fit(X1_train, y1_train)

# Best model from RandomizedSearchCV
reg_best_model = reg_search.best_estimator_

# Predict and evaluate
y1_pred = reg_best_model.predict(X1_test)
rmse = np.sqrt(mean_squared_error(y1_test, y1_pred))
print(f"Model 1 (Sales Prediction) RMSE: {rmse:.2f}")

# Save regression model
with open('models/sales_model1.pkl2', 'wb') as f:
    pickle.dump(reg_best_model, f)

# ---------------------
# Model 2: Profit Classification (with XGBoost and SMOTE)
# ---------------------
df['profit_class'] = (df['profit_per_order'] > 0).astype(int)
X2 = X1  # same features
y2 = df['profit_class']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Apply SMOTE for class balancing
smote = SMOTE(random_state=42)
X2_train_smote, y2_train_smote = smote.fit_resample(X2_train, y2_train)

# Define the parameter grid for RandomizedSearchCV
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.7, 0.8, 0.9, 1],
    'colsample_bytree': [0.7, 0.8, 0.9, 1],
    'n_estimators': [100, 200, 300]
}

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)

# RandomizedSearchCV to optimize XGBoost
xgb_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid_xgb, n_iter=10, cv=3, verbose=1, n_jobs=-1, random_state=42)
xgb_search.fit(X2_train_smote, y2_train_smote)

# Best parameters from XGBoost
print("Best parameters found for XGBoost: ", xgb_search.best_params_)

# Use the best XGBoost model
xgb_best_model = xgb_search.best_estimator_

# Predict and evaluate
y2_pred = xgb_best_model.predict(X2_test)
accuracy = accuracy_score(y2_test, y2_pred)
f1 = f1_score(y2_test, y2_pred)
roc_auc = roc_auc_score(y2_test, xgb_best_model.predict_proba(X2_test)[:, 1])

print(f"Optimized Model 2 (Profit Classification) Accuracy: {accuracy:.2f}")
print(f"Optimized Model 2 (F1 Score): {f1:.2f}")
print(f"Optimized Model 2 (ROC-AUC Score): {roc_auc:.2f}")

# Save classification model
with open('models/profit_model2.pkl', 'wb') as f:
    pickle.dump(xgb_best_model, f)

# Save encoders
with open('models/label_encoders2.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Models and encoders saved in 'models/' folder.")
