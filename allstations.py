# print_all_stations.py

import pandas as pd

# Load the dataset
df = pd.read_csv("railway_cleaned.csv")

# Ensure the required columns exist
required_cols = ['Departure Station', 'Arrival Destination']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f" Column '{col}' not found in the dataset.")

# Combine both departure and arrival station columns into one Series
all_stations = pd.concat([df['Departure Station'], df['Arrival Destination']])

# Drop missing values and get unique station names
unique_stations = sorted(all_stations.dropna().unique())


print(f"\n Total unique stations in the dataset: {len(unique_stations)}\n")
print(" All unique station names:\n")
for station in unique_stations:
    print(station)

# total_journeys.py

import pandas as pd

# Load the dataset
df = pd.read_csv("railway_cleaned.csv")

# Check required columns
required_cols = ['Departure Station', 'Arrival Destination', 'Date of Journey']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f" Column '{col}' not found in the dataset.")

# Count total number of journeys (each row = one journey)
total_journeys = len(df)

# Display result
print(f"\n Total number of journeys in the dataset: {total_journeys}")

# journeys_per_month.py

import pandas as pd

# Load the dataset
df = pd.read_csv("railway_cleaned.csv")

# Ensure 'Date of Journey' exists
if 'Date of Journey' not in df.columns:
    raise ValueError("Column 'Date of Journey' not found in the dataset.")

# Convert to datetime
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=['Date of Journey'])

# Extract year and month
df['YearMonth'] = df['Date of Journey'].dt.to_period('M')

# Count number of journeys per month
monthly_counts = df['YearMonth'].value_counts().sort_index()

# Display results
print("\n Number of journeys per month:\n")
for month, count in monthly_counts.items():
    print(f"{month}: {count}")

# Optional: Save to CSV
monthly_counts.to_csv("journeys_per_month.csv", header=["Journeys"])
print("\n Saved summary to 'journeys_per_month.csv'")
o
# journey_length_per_month.py

import pandas as pd

# Load dataset
df = pd.read_csv("railway_cleaned.csv")

# Identify the correct distance column
possible_columns = ['Distance', 'Distance (km)', 'Journey Length']
distance_col = next((col for col in possible_columns if col in df.columns), None)

if not distance_col:
    raise ValueError(" No distance column found in the dataset.")

# Convert 'Date of Journey' to datetime
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df = df.dropna(subset=['Date of Journey', distance_col])

# Remove negative or invalid distances (optional)
df = df[df[distance_col] > 0]

# Extract month-year
df['YearMonth'] = df['Date of Journey'].dt.to_period('M')

# Calculate total journey length per month
monthly_lengths = df.groupby('YearMonth')[distance_col].sum()

# Display results
print("\n Total journey length per month:\n")
for month, length in monthly_lengths.items():
    print(f"{month}: {length:.2f} km")


