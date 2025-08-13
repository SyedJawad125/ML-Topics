from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    "Annual_Income": [15, 16, 17, 30, 45, 46, 47, 70, 80, 85],
    "Spending_Score": [39, 81, 6, 77, 40, 76, 15, 10, 85, 20]
}
df = pd.DataFrame(data)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
df['Cluster'] = kmeans.fit_predict(df[['Annual_Income', 'Spending_Score']])

print(df)

# Plot clusters
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segments")
plt.show()
