# # Importing necessary libraries
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix

# # Step 3: Create the dataset
# data = {
#     'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of study hours
#     'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Whether the student passed (0: Fail, 1: Pass)
# }

# # Step 4: Load the data into a pandas DataFrame
# df = pd.DataFrame(data)

# # Step 5: Define the independent (X) and dependent (Y) variables
# X = df[['StudyHours']]  # Feature matrix (independent variable)
# Y = df['Passed']  # Target variable (dependent variable)

# # Step 6: Split the data into training and testing sets
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Step 7: Create the Logistic Regression model and train it
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # Step 8: Predict outcomes on the test set
# Y_pred = model.predict(X_test)

# # Step 9: Print out the accuracy of the model
# accuracy = accuracy_score(Y_test, Y_pred)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# # Step 10: Print the confusion matrix
# conf_matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
# print("Confusion Matrix:")
# print(conf_matrix)

# # Step 11: Visualize the data and decision boundary
# plt.scatter(X, Y, color='blue')  # Plot the actual data points
# plt.plot(X, model.predict_proba(X)[:, 1], color='red')  # Plot the predicted probabilities
# plt.title('Study Hours vs. Probability of Passing')
# plt.xlabel('Study Hours')
# plt.ylabel('Probability of Passing')
# plt.show()

# # Step 12: Test the model by predicting whether a student who studied for 4.5 hours will pass
# new_data = pd.DataFrame([[4.5]], columns=['StudyHours'])
# predicted_outcome = model.predict(new_data)
# predicted_probability = model.predict_proba(new_data)[:, 1]

# print(f"Predicted outcome for 4.5 study hours: {'Pass' if predicted_outcome[0] == 1 else 'Fail'}")
# print(f"Predicted probability of passing: {predicted_probability[0]:.2f}")



# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 3: Create the dataset
data = {
    'StudyHours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Number of study hours
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Whether the student passed (0: Fail, 1: Pass)
}

# Step 4: Load the data into a pandas DataFrame
df = pd.DataFrame(data)

# Step 5: Define the independent (X) and dependent (Y) variables
X = df[['StudyHours']]  # Feature matrix (independent variable)
Y = df['Passed']  # Target variable (dependent variable)

# Step 6: Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 7: Create the Logistic Regression model and train it
model = LogisticRegression()
model.fit(X_train, Y_train)

# Step 8: Predict outcomes on the test set
Y_pred = model.predict(X_test)

# Step 9: Print out the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 10: Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
print("Confusion Matrix:")
print(conf_matrix)

# Step 11: Visualize the data and decision boundary
plt.scatter(X, Y, color='blue')  # Plot the actual data points
plt.plot(X, model.predict_proba(X)[:, 1], color='red')  # Plot the predicted probabilities
plt.title('Study Hours vs. Probability of Passing')
plt.xlabel('Study Hours')
plt.ylabel('Probability of Passing')
plt.show()

# Step 12: Take user input for study hours and predict
user_hours = float(input("Enter study hours: "))
new_data = pd.DataFrame([[user_hours]], columns=['StudyHours'])
predicted_outcome = model.predict(new_data)
predicted_probability = model.predict_proba(new_data)[:, 1]

print(f"Predicted outcome for {user_hours} study hours: {'Pass' if predicted_outcome[0] == 1 else 'Fail'}")
print(f"Predicted probability of passing: {predicted_probability[0]:.2f}")
