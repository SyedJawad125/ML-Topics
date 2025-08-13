from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

y_true = [1, 0, 1, 1, 0, 0]
y_pred = [1, 0, 1, 0, 0, 1]
y_probs = [0.9, 0.2, 0.8, 0.4, 0.3, 0.7]  # predicted probabilities for positive class

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("ROC-AUC:", roc_auc_score(y_true, y_probs))
