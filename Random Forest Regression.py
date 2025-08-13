# # Import necessary libraries
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # Create a synthetic dataset
# data = {
#     'bedrooms': np.random.randint(1, 5, size=1000),
#     'bathrooms': np.random.randint(1, 3, size=1000),
#     'kitchen': np.random.randint(1, 2, size=1000),
#     'size': np.random.randint(500, 2000, size=1000),  # size in sq.ft.
#     'price': np.random.randint(10, 40, size=1000) * 100000  # price in lakh
# }

# # Convert into DataFrame
# df = pd.DataFrame(data)

# # Define the feature set and the target variable
# X = df[['bedrooms', 'bathrooms', 'kitchen', 'size']]
# y = df['price']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a Random Forest Regressor model
# model = RandomForestRegressor(n_estimators=100, random_state=42)

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)

# print(f"Root Mean Squared Error: {rmse}")

# # To predict for a new query
# new_flat = [[5, 4, 2, 2000]]  # Example: 3 bedrooms, 2 bathrooms, 1 kitchen, size=1200 sq.ft
# predicted_price = model.predict(new_flat)
# print(f"Predicted price for the flat: {predicted_price[0]}")




# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 1. Load dataset from CSV
df = pd.read_csv(r"F:\datasets\housing_data.csv")  # make sure this file is in the same directory

# 2. Define the feature set (X) and the target variable (y)
X = df[['bedrooms', 'bathrooms', 'kitchen', 'size']]
y = df['price']

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Make predictions for test set
y_pred = model.predict(X_test)

# 7. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

# 8. Take user input for a new flat prediction
print("\nEnter flat details to predict price:")
bedrooms = int(input("Number of bedrooms: "))
bathrooms = int(input("Number of bathrooms: "))
kitchen = int(input("Number of kitchens: "))
size = int(input("Size in sq.ft: "))

#9 Create DataFrame with same column names as training
input_data = pd.DataFrame([[bedrooms, bathrooms, kitchen, size]],
                          columns=['bedrooms', 'bathrooms', 'kitchen', 'size'])

#10 Predict
predicted_price = model.predict(input_data)[0]
print(f"Predicted price for the flat: {predicted_price:,.0f}")
