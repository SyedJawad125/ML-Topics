# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 3: Create the dataset
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8],  # Age of the car in years
    'Price': [45, 42, 38, 35, 30, 25, 20, 18]  # Price of the car in $1000s
}

# Step 4: Load the data into a pandas DataFrame
df = pd.DataFrame(data)

# Step 5: Define the independent (X) and dependent (Y) variables
X = df[['Age']]  # Feature matrix (independent variable)
Y = df['Price']  # Target variable (dependent variable)

# Step 6: Transform the features into polynomial features
# Create polynomial features of degree 2 (quadratic)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Step 7: Create the Polynomial Regression model and train it
model = LinearRegression()
model.fit(X_poly, Y)

# Step 8: Predict outcomes
Y_pred = model.predict(X_poly)

# Step 9: Print out the coefficients and intercept
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

# Step 10: Visualize the data and the polynomial regression curve
plt.scatter(X, Y, color='blue')  # Plot the actual data points
plt.plot(X, Y_pred, color='red')  # Plot the polynomial regression curve
plt.title('Car Age vs. Price')
plt.xlabel('Age (years)')
plt.ylabel('Price ($1000s)')
plt.show()

# Step 11: Predict the price of a car that is 5.5 years old
predicted_price = model.predict(poly.transform([[5.5]]))
print(f"Predicted price for a 5.5-year-old car: ${predicted_price[0]:.2f} thousand")
