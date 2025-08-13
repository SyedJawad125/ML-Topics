import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------
# 1. Create a sample dataset
# -------------------------------
data = np.array([10, 12, 15, 15, 17, 21, 21, 21, 23, 30])

df = pd.DataFrame({"Values": data})
print("Dataset:\n", df)

# -------------------------------
# 2. Mean, Median, Mode
# -------------------------------
mean_val = np.mean(data)
median_val = np.median(data)
mode_val = stats.mode(data, keepdims=True)  # returns Mode and Count

print("\nMean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val.mode[0], "Count:", mode_val.count[0])

# -------------------------------
# 3. Variance & Standard Deviation
# -------------------------------
variance_val = np.var(data, ddof=1)  # ddof=1 → sample variance
std_dev_val = np.std(data, ddof=1)   # sample std deviation

print("\nVariance:", variance_val)
print("Standard Deviation:", std_dev_val)

# -------------------------------
# 4. Probability basics
# Example: Probability of getting a number > 20
# -------------------------------
prob_gt_20 = np.sum(data > 20) / len(data)
print("\nProbability(Value > 20):", prob_gt_20)

# -------------------------------
# 5. Normal Distribution
# -------------------------------
# Generate 1000 random numbers from a normal distribution
normal_data = np.random.normal(loc=50, scale=10, size=1000)

# Plot histogram
plt.hist(normal_data, bins=30, color='skyblue', edgecolor='black', density=True)
plt.title("Normal Distribution (mean=50, std=10)")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 6. Correlation & Covariance
# -------------------------------
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

correlation = np.corrcoef(x, y)[0, 1]
covariance = np.cov(x, y)[0, 1]

print("\nCorrelation between x and y:", correlation)
print("Covariance between x and y:", covariance)
