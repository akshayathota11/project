import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

df = pd.read_csv("railway_cleaned.csv")
print("Dataset Shape:", df.shape)
print("\nColumn Info:")
print(df.info())
#ststistics
print("\nSummary Statistics:")
print(df.describe(include='all'))
print("\nJourney Status Count:")
print(df['Journey Status'].value_counts())
print("\nTicket Type Distribution:")
print(df['Ticket Type'].value_counts())

#reason for the delay
unique_reasons = df['Reason for Delay'].nunique()
print("Number of unique delay reasons:", unique_reasons)
print("Delay Reason Types:")
print(df['Reason for Delay'].unique())
print("Delay Reasons with Counts:")
print(df['Reason for Delay'].value_counts())
delayed_df = df[df['Journey Status'] == 'Delayed']
print(delayed_df['Reason for Delay'].value_counts())

#delay rate by type 
delay_rate_by_type = df.groupby('Ticket Type')['Journey Status'].value_counts(normalize=True).unstack()
print(delay_rate_by_type)

#delay rate bt ticket type
delay_rate_by_type.plot(kind='bar', stacked=True, figsize=(8, 5))
plt.title("Delay Rate by Ticket Type")
plt.ylabel("Proportion of Journeys")
plt.xlabel("Ticket Type")
plt.legend(title='Journey Status')
plt.tight_layout()
plt.savefig("delay_rate_by_ticket_type.png")
plt.show()

# Convert Departure Time to datetime format
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')

# hour of a day analysis
plt.figure(figsize=(10, 5))
sns.countplot(data=df[df['Journey Status'] == 'Delayed'], x='Hour')
plt.title("Delays by Hour of Day")
plt.xlabel("Hour of Departure")
plt.ylabel("Number of Delayed Trains")
plt.tight_layout()
plt.savefig("delays_by_hour.png")
plt.show()

# Plot delay rate by day of week
sns.barplot(data=df, x='Day of Week', y=(df['Journey Status'] == 'Delayed').astype(int))
plt.title("Delay Rate by Day of Week")
plt.ylabel("Proportion Delayed")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#distribution for ticket type
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Ticket Type', order=df['Ticket Type'].value_counts().index)
plt.title('Ticket Type Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("ticket_type_distribution.png")
plt.show()




# converting Date of Journey is datetime
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Day'] = df['Date of Journey'].dt.day_name()

# Treemap
plt.figure(figsize=(12, 7))
squarify.plot(sizes=top_reasons.values, label=top_reasons.index, alpha=0.8, pad=True)
plt.title("Treemap of Top 10 Reasons for Delay")
plt.axis('off')
plt.tight_layout()
plt.savefig("delay_reason_treemap.png")
plt.show()

# Filtering delays
delayed = df[df['Journey Status'] == 'Delayed']

# pivot table: Reason vs Day
heatmap_data = delayed.pivot_table(index='Reason for Delay', columns='Day', aggfunc='size', fill_value=0)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data.reindex(columns=day_order, fill_value=0)

#heatmap for dalay reasons
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d")
plt.title("Heatmap of Delay Reasons by Day of the Week")
plt.xlabel("Day")
plt.ylabel("Reason for Delay")
plt.tight_layout()
plt.savefig("delay_reason_heatmap.png")
plt.show()
