from sklearn.model_selection import GridSearchCV

params = {"max_depth": [2, 3, 4], "min_samples_split": [2, 3, 4]}
grid = GridSearchCV(DecisionTreeClassifier(), params, cv=3)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)
