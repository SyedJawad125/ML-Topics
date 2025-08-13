from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({
    "Category": ["A", "B", "C", "A", "B", "C"],
    "Values": [4, 7, 3, 8, 5, 6],
    "Score": [70, 82, 90, 65, 88, 75]
})

# Bar plot
sns.barplot(x="Category", y="Values", data=df)

# Scatter plot with regression line
sns.lmplot(x="Values", y="Score", data=df)

# Heatmap example
matrix = np.random.rand(4, 4)
sns.heatmap(matrix, annot=True, cmap="coolwarm")
plt.show()
