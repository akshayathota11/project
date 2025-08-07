import pandas as pd

# Load your dataset
df = pd.read_csv("railway_cleaned.csv")

# Normalize "Reason for Delay"
df['Reason for Delay'] = df['Reason for Delay'].str.strip()       # Remove extra spaces
df['Reason for Delay'] = df['Reason for Delay'].str.title()       # Capitalize properly
df['Reason for Delay'] = df['Reason for Delay'].replace({
    'Signal failure': 'Signal Failure'  # Merge case variations
})

# Save the cleaned file
df.to_csv("railway_cleaned1.csv", index=False)

print(" Delay reasons cleaned and saved back to railway_cleaned.csv")
print("Unique delay reasons now:", df['Reason for Delay'].unique())

