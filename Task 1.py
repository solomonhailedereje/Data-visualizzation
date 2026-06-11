#Taks 1

import pandas as pd

df = pd.read_csv('merged_dataset.csv')

# Fix Rainfall_mm type
df['Rainfall_mm'] = pd.to_numeric(df['Rainfall_mm'], errors='coerce')

# Report missing values
print("Missing values per column:")
print(df.isnull().sum())
print(f"\nTotal rows before cleaning: {len(df)}")

# Drop rows with missing values
df = df.dropna()

print(f"Total rows after cleaning: {len(df)}")
print(f"Rows removed: {55375 - len(df)}")

