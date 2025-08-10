import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

df = pd.read_csv("railway_cleaned.csv")
print("Dataset Shape:", df.shape)

print("\nColumn Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all'))

print("\nJourney Status Count:")
print(df['Journey Status'].value_counts())

print("\nTicket Type Distribution:")
print(df['Ticket Type'].value_counts())

# --- Standardize / prepare time columns ---
df['Date of Journey']   = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time']    = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')

# Create features AFTER conversion
df['Hour'] = df['Departure Time'].dt.hour
df['Day']  = df['Date of Journey'].dt.day_name()

# Drop rows where Hour is missing for hour-based plots
df_hour_ok = df.dropna(subset=['Hour']).copy()
df_hour_ok['Hour'] = df_hour_ok['Hour'].astype(int)

# --- Delay reasons summary ---
print("\nNumber of unique delay reasons:", df['Reason for Delay'].nunique())
print("Delay Reason Types:")
print(df['Reason for Delay'].unique())

print("Delay Reasons with Counts:")
reason_counts = df['Reason for Delay'].value_counts()
print(reason_counts)

# Delayed-only counts (exclude "No Delay")
delayed_df = df[df['Journey Status'] == 'Delayed'].copy()
print(delayed_df['Reason for Delay'].value_counts())

# --- Delay rate by ticket type ---
delay_rate_by_type = (
    df.groupby('Ticket Type')['Journey Status']
      .value_counts(normalize=True)
      .unstack()
      .fillna(0)
)
print("\nDelay rate by ticket type:")
print(delay_rate_by_type)

plt.figure(figsize=(8,5))
delay_rate_by_type.plot(kind='bar', stacked=True)
plt.title("Delay Rate by Ticket Type")
plt.ylabel("Proportion of Journeys")
plt.xlabel("Ticket Type")
plt.legend(title='Journey Status')
plt.tight_layout()
plt.savefig("delay_rate_by_ticket_type.png")
plt.close()

# --- Delays by hour of day (use df_hour_ok to ensure Hour exists) ---
plt.figure(figsize=(10,5))
sns.countplot(data=df_hour_ok[df_hour_ok['Journey Status']=='Delayed'], x='Hour', order=list(range(24)))
plt.title("Delays by Hour of Day")
plt.xlabel("Hour of Departure")
plt.ylabel("Number of Delayed Trains")
plt.tight_layout()
plt.savefig("delays_by_hour.png")
plt.close()

# --- Delay rate by day of week ---
# Build a binary delayed flag
df['is_delayed'] = (df['Journey Status'] == 'Delayed').astype(int)
day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

day_delay = df.groupby('Day', dropna=False)['is_delayed'].mean().reindex(day_order)
plt.figure(figsize=(9,4))
sns.barplot(x=day_delay.index, y=day_delay.values)
plt.title("Delay Rate by Day of Week")
plt.ylabel("Proportion Delayed")
plt.xlabel("Day")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("delay_rate_by_day.png")
plt.close()

# --- Treemap of top 10 delay reasons (exclude 'No Delay') ---
top_reasons = (delayed_df['Reason for Delay']
               .value_counts()
               .head(10))

plt.figure(figsize=(12,7))
squarify.plot(sizes=top_reasons.values,
              label=[f"{k}\n{v}" for k,v in top_reasons.items()],
              alpha=0.85, pad=True)
plt.title("Treemap of Top 10 Reasons for Delay")
plt.axis('off')
plt.tight_layout()
plt.savefig("delay_reason_treemap.png")
plt.close()

# --- Heatmap of delay reasons by day ---
heatmap_data = (delayed_df
                .assign(Day=delayed_df['Date of Journey'].dt.day_name())
                .pivot_table(index='Reason for Delay', columns='Day', aggfunc='size', fill_value=0)
                .reindex(columns=day_order, fill_value=0))

plt.figure(figsize=(14,8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Heatmap of Delay Reasons by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Reason for Delay")
plt.tight_layout()
plt.savefig("delay_reason_heatmap.png")
plt.close()

print("\n EDA visuals saved:")
print(" - delay_rate_by_ticket_type.png")
print(" - delays_by_hour.png")
print(" - delay_rate_by_day.png")
print(" - delay_reason_treemap.png")
print(" - delay_reason_heatmap.png")

