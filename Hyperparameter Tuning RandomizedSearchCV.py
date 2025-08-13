# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np

# params = {
#     "max_depth": np.arange(2, 10),
#     "min_samples_split": np.arange(2, 10)
# }
# rand = RandomizedSearchCV(DecisionTreeClassifier(), params, cv=3, n_iter=5, random_state=42)
# rand.fit(X_train, y_train)

# print("Best Parameters:", rand.best_params_)



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1️⃣ Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------
# Model 1: Without tuning
# --------------------------
model_default = DecisionTreeClassifier()  # uses default parameters
model_default.fit(X_train, y_train)
y_pred_default = model_default.predict(X_test)
default_acc = accuracy_score(y_test, y_pred_default)

# --------------------------
# Model 2: With GridSearchCV tuning
# --------------------------
params = {"max_depth": [2, 3, 4, 5], "min_samples_split": [2, 3, 4]}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=3)
grid.fit(X_train, y_train)
y_pred_tuned = grid.predict(X_test)
tuned_acc = accuracy_score(y_test, y_pred_tuned)

# --------------------------
# Results
# --------------------------
print("---- Model Performance ----")
print(f"Without Tuning Accuracy: {default_acc:.4f}")
print(f"With Tuning Accuracy:    {tuned_acc:.4f}")
print(f"Best Parameters from GridSearchCV: {grid.best_params_}")
