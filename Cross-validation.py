from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# 1️⃣ Load sample dataset
iris = load_iris()
X = iris.data       # Features
y = iris.target     # Labels

# 2️⃣ Define model
clf = DecisionTreeClassifier()

# 3️⃣ Perform 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)





from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Model
clf = DecisionTreeClassifier()

# 5-fold cross-validation
scores = cross_val_score(clf, X, y, cv=5)  # cv=5 means 5 folds
print("Scores for each fold:", scores)
print("Average score:", scores.mean())
