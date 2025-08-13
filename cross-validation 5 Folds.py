from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# 1️⃣ Load sample dataset (Iris dataset)
X, y = load_iris(return_X_y=True)

# 2️⃣ Create a Decision Tree model (default parameters)
clf = DecisionTreeClassifier()

# 3️⃣ Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)  # cv=5 means 5 folds

# 4️⃣ Show results
print("Scores for each fold:", scores)
print("Average cross-validation score:", scores.mean())
print("Standard deviation of scores:", scores.std())

# Optional: Train once on the full data and check accuracy
clf.fit(X, y)
print("Training accuracy (on full dataset):", clf.score(X, y))



# output

# Scores for each fold: [0.9667 0.9667 0.9    0.9667 1.0   ]
# Average cross-validation score: 0.96
# Standard deviation of scores: 0.0351
# Training accuracy (on full dataset): 1.0
