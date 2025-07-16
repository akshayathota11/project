import pandas as pd
df = pd.read_csv('railway system dataset.csv')
print(df.head())
print("\nColumns:", df.columns.tolist())
df['Railcard'] = df['Railcard'].fillna('No Railcard')
df['Reason for Delay'] = df['Reason for Delay'].fillna('No Delay')
df = df.dropna(subset=['Actual Arrival Time'])
print("\nMissing values:\n", df.isnull().sum())
df.to_csv('railway_cleaned.csv', index=False)



