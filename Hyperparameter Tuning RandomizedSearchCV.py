from sklearn.model_selection import RandomizedSearchCV
import numpy as np

params = {
    "max_depth": np.arange(2, 10),
    "min_samples_split": np.arange(2, 10)
}
rand = RandomizedSearchCV(DecisionTreeClassifier(), params, cv=3, n_iter=5, random_state=42)
rand.fit(X_train, y_train)

print("Best Parameters:", rand.best_params_)
