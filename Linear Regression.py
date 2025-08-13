# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 3: Create the dataset
# data = {
#     'Size': [800, 1000, 1200, 1500, 1800],  # Size of the house (sq ft)
#     'Price': [200000, 250000, 300000, 375000, 450000]  # Price of the house ($)
# }
df = pd.read_csv(r"F:\datasets\house_prices.txt")



# Step 4: Load the data into a pandas DataFrame
# df = pd.DataFrame(data)

# Step 5: Define the independent (X) and dependent (Y) variables
X = df[['Size']]  # Feature matrix (independent variable)
Y = df['Price']  # Target variable (dependent variable)

# Step 6: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 7: Create the Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, Y_train)

# Step 8: Predict prices using the trained model
Y_pred = model.predict(X_test)

# Step 9: Print out the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Step 10: Visualize the results
plt.scatter(X, Y, color='blue')  # Plot the actual data points
plt.plot(X, model.predict(X), color='red')  # Plot the regression line
plt.title('House Size vs. Price')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.show()
# Step 11: Test the model by predicting the price of a 1400 sq ft house
# Step 11: Take size from user and predict the price
size_input = float(input("Enter the house size in sq ft: "))
predicted_price = model.predict([[size_input]])
print(f"Predicted price for a {size_input} sq ft house: ${predicted_price[0]:.2f}")

