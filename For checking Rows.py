import pandas as pd

# Read CSV
# df = pd.read_csv("flats.csv")
df = pd.read_csv(r"F:\datasets\house_prices.txt")
df.dropna(inplace=True)  # Removes any rows with missing values

# Read Excel (requires openpyxl or xlrd installed)
# df = pd.read_excel("flats.xlsx")

# Check first few rows
print(df.head())

# Check basic info
print(df.info())


print(df.isnull().sum())

# Percentage of missing values
print(df.isnull().mean() * 100)