#for all delayes trains reasons
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
print(f"\n Total delayed journeys: {len(all_delayed)}\n")
print("🎯 Sample of delayed journeys:")
print(all_delayed[['Route', 'Date of Journey', 'Departure Time', 'Ticket Type']].head(20))

# Step 6: Export to CSV (optional)
output_path = "all_delayed_trains.csv"
all_delayed.to_csv(output_path, index=False)
print(f"\n📁 Full list of delayed journeys saved to: {output_path}")


#pie chart for all delayed trains
# piechart_all_delayed_reasons.py

import pandas as pd
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv("railway_cleaned.csv")

# Step 1: Create Is_Delayed column
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if str(x).lower() == 'delayed' else 0)

# Step 2: Detect delay reason column
reason_col_candidates = ['Reason for Delay', 'Delay Reason', 'Reason']
reason_col = next((col for col in reason_col_candidates if col in df.columns), None)

if reason_col is None:
    raise ValueError(" Could not find a 'Reason for Delay' column in the dataset.")

# Step 3: Filter only delayed journeys
delayed_df = df[df['Is_Delayed'] == 1]

# Step 4: Count delay reasons
reason_counts = delayed_df[reason_col].value_counts()

# Step 5: Optionally group smaller categories into "Others"
top_n = reason_counts.head(6)
top_n['Others'] = reason_counts.iloc[6:].sum()

# Step 6: Create pie chart
plt.figure(figsize=(8, 8))
top_n.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel2.colors)
plt.title("Delay Reasons for All Delayed Trains")
plt.ylabel("")
plt.tight_layout()

# Step 7: Save chart
os.makedirs("visuals", exist_ok=True)
output_path = "visuals/all_delay_reasons_pie.png"
plt.savefig(output_path)
plt.show()

print(f"\n Pie chart saved to: {output_path}")

#code for all repeated delayed trains
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("railway_cleaned.csv")

# Step 1: Create the 'Is_Delayed' column if not already present
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Step 2: Create a 'Route' column for grouping
df['Route'] = df['Departure Station'] + " → " + df['Arrival Destination']

# Optional: Convert date if you want per-day or per-week analysis
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')

# Step 3: Filter only delayed journeys
delayed_df = df[df['Is_Delayed'] == 1]

# Step 4: Count delays per Route
route_delay_counts = delayed_df.groupby('Route').size()

# Step 5: Filter to find routes delayed multiple times
repeated_route_delays = route_delay_counts[route_delay_counts > 1]
print(f" Total number of routes with multiple delays: {len(repeated_route_delays)}\n")

# Show top 10 routes with the most repeated delays
print(" All repeatedly delayed routes:")
print(repeated_route_delays.sort_values(ascending=False).head(20))

# Step 6: Visualize the top 20 routes
top_20 = repeated_route_delays.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
top_20.plot(kind='barh', color='tomato')
plt.title("All Routes with Most Repeated Delays")
plt.xlabel("Number of Delays")
plt.ylabel("Route")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("repeated_route_delays.png")
plt.show()




