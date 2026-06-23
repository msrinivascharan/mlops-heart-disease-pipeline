import pandas as pd

# Column names for this dataset
columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Load data
df = pd.read_csv('data/processed.cleveland.data',
                 names=columns, na_values='?')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic stats:")
print(df.describe())
print("\nTarget distribution:")
print(df['target'].value_counts())

# Convert target to binary: 0 = no disease, 1 = disease
df['target'] = (df['target'] > 0).astype(int)

# Drop rows with missing values (only 6 rows — safe to drop)
df = df.dropna()

# Save clean version
df.to_csv('data/heart_clean.csv', index=False)
print("\nClean data saved to data/heart_clean.csv")
print("Clean data shape:", df.shape)
