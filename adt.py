# all_delayed_trains.py

import pandas as pd
import os

# Load your cleaned dataset
df = pd.read_csv("railway_cleaned.csv")

# Step 1: Create a binary column for delay (if not already present)
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if str(x).lower() == 'delayed' else 0)

# Step 2: Combine departure and arrival stations into a route name
df['Route'] = df['Departure Station'] + " → " + df['Arrival Destination']

# Step 3: Convert dates and times (optional but useful)
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')

# Step 4: Identify all delayed trains
all_delayed = df[df['Is_Delayed'] == 1]

# Step 5: Display sample of delayed trains
print(f"\n✅ Total delayed journeys: {len(all_delayed)}\n")
print("🎯 Sample of delayed journeys:")
print(all_delayed[['Route', 'Date of Journey', 'Departure Time', 'Ticket Type']].head(20))

# Step 6: Export to CSV (optional)
output_path = "all_delayed_trains.csv"
all_delayed.to_csv(output_path, index=False)
print(f"\n📁 Full list of delayed journeys saved to: {output_path}")

