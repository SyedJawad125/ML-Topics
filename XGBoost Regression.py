# # Install if not already installed
# # pip install xgboost scikit-learn pandas numpy

# import pandas as pd
# import numpy as np
# from xgboost import XGBRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # ------------------------------
# # 1. Create a sample dataset
# # ------------------------------
# # data = {
# #     "Bedrooms": [2, 3, 4, 3, 5, 4, 2, 6, 3, 4],
# #     "Bathrooms": [1, 2, 3, 2, 4, 3, 1, 5, 2, 3],
# #     "Size_sqft": [850, 1200, 2000, 1500, 3000, 2200, 900, 3500, 1600, 2500],
# #     "Age": [5, 10, 20, 15, 5, 8, 25, 3, 12, 7],
# #     "Price": [150000, 200000, 350000, 250000, 500000, 400000, 160000, 600000, 270000, 450000]
# # }

# # df = pd.DataFrame(data)

# # Load dataset from file
# df = pd.read_csv(r"F:\datasets\housing_dataset_xgboost.csv")  # Make sure this CSV is in the same folder as your script

# # Features & Target
# X = df[["Bedrooms", "Bathrooms", "Size_sqft", "Age"]]
# y = df["Price"]

# # ------------------------------
# # 2. Train-Test Split
# # ------------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ------------------------------
# # 3. Initialize & Train XGBoost Model
# # ------------------------------
# model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
# model.fit(X_train, y_train)

# # ------------------------------
# # 4. Predictions & Evaluation
# # ------------------------------
# y_pred = model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print(f"Root Mean Squared Error: {rmse:,.2f}")

# # ------------------------------
# # 5. Predict New House Price
# # ------------------------------
# print("\nEnter new house details to predict price:")
# bedrooms = int(input("Bedrooms: "))
# bathrooms = int(input("Bathrooms: "))
# size = int(input("Size in sq.ft: "))
# age = int(input("Age of house: "))

# pred_price = model.predict([[bedrooms, bathrooms, size, age]])[0]
# print(f"\nPredicted Price: ${pred_price:,.0f}")



import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- 1. Load data and train model ---
df = pd.read_csv(r"F:\datasets\housing_dataset_xgboost.csv")   # make sure this file is next to the script

X = df[['Bedrooms', 'Bathrooms', 'Size_sqft', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse:,.2f}\n")

# --- 2. Inspect dataset and training ranges ---
print("Dataset summary (first rows):")
print(df.head(), "\n")

print("Feature statistics:")
print(df.describe().T[['min', 'max', 'mean', '50%']], "\n")

# Training ranges (use X_train to reflect what model saw)
train_min = X_train.min()
train_max = X_train.max()
print("Training feature ranges (min / max):")
for col in X_train.columns:
    print(f"  {col}: {train_min[col]}  ->  {train_max[col]}")
print()

# Feature importances
fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature importances:")
print(fi, "\n")

# --- 3. Get user input and show raw vs capped prediction ---
print("Enter new house details to predict price:")
bedrooms = int(input("Bedrooms: "))
bathrooms = int(input("Bathrooms: "))
size = int(input("Size in sq.ft: "))
age = int(input("Age of house: "))

# Build DataFrame with same column names
input_df = pd.DataFrame([[bedrooms, bathrooms, size, age]],
                        columns=['Bedrooms', 'Bathrooms', 'Size_sqft', 'Age'])

# Cap inputs to training min/max to avoid extrapolation (you can remove capping if you want raw)
capped_df = input_df.copy()
for col in input_df.columns:
    capped_df[col] = capped_df[col].clip(lower=train_min[col], upper=train_max[col])

# Predictions
raw_pred = model.predict(input_df)[0]
capped_pred = model.predict(capped_df)[0]

print("\n--- Results ---")
print(f"Raw input: {input_df.to_dict(orient='records')[0]}")
print(f"Prediction (raw input)   : ${raw_pred:,.0f}")
print(f"Capped to train ranges   : {capped_df.to_dict(orient='records')[0]}")
print(f"Prediction (capped input) : ${capped_pred:,.0f}")

# Helpful diagnostic: price per sqft for capped input
if capped_df['Size_sqft'].values[0] > 0:
    print(f"Predicted price per sqft (capped): ${capped_pred / capped_df['Size_sqft'].values[0]:,.2f}")
