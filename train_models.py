import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score

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
# Model 1: Sales Prediction
# ---------------------
X1 = df[features]
y1 = df['sales_per_order']

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
reg_model.fit(X1_train, y1_train)
y1_pred = reg_model.predict(X1_test)
print(f"Model 1 (Regression) RMSE: {np.sqrt(mean_squared_error(y1_test, y1_pred)):.2f}")

# Save regression model
with open('models/sales_model.pkl', 'wb') as f:
    pickle.dump(reg_model, f)

# ---------------------
# Model 2: Profit Classification
# ---------------------
df['profit_class'] = (df['profit_per_order'] > 0).astype(int)
X2 = X1  # same features
y2 = df['profit_class']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X2_train, y2_train)
y2_pred = clf_model.predict(X2_test)
print(f"Model 2 (Classification) Accuracy: {accuracy_score(y2_test, y2_pred):.2f}")

# Save classification model
with open('models/profit_model.pkl', 'wb') as f:
    pickle.dump(clf_model, f)

# Save encoders
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Models and encoders saved in 'models/' folder.")
