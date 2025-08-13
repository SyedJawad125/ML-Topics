import matplotlib.pyplot as plt
import numpy as np

# Bar Chart
categories = ['A', 'B', 'C', 'D']
values = [3, 7, 5, 9]
plt.bar(categories, values, color='blue')
plt.title("Bar Chart Example")
plt.show()

# Scatter Plot
x = [5, 7, 8, 7, 6, 9, 5, 4, 6, 7]
y = [99, 86, 87, 88, 100, 86, 103, 87, 94, 78]
plt.scatter(x, y, color='red')
plt.title("Scatter Plot Example")
plt.show()

# Histogram
data = np.random.randn(1000)  # 1000 random values
plt.hist(data, bins=20, color='green')
plt.title("Histogram Example")
plt.show()
