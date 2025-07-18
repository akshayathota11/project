# predict_delay_probability.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("railway_cleaned.csv")

# Convert to datetime
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')

# Feature engineering
df['Hour'] = df['Departure Time'].dt.hour
df['Day'] = df['Date of Journey'].dt.day_name()
df['Route'] = df['Departure Station'] + " → " + df['Arrival Destination']
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Select features
features = ['Route', 'Ticket Class', 'Ticket Type', 'Purchase Type',
            'Payment Method', 'Hour', 'Day', 'Price']
X = pd.get_dummies(df[features])
y = df['Is_Delayed']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Evaluation
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ ROC AUC Score:", roc_auc_score(y_test, y_proba > 0.5))

# Top 10 high-risk predictions
results = X_test.copy()
results['Actual'] = y_test
results['Predicted_Probability'] = y_proba
top_predictions = results.sort_values(by='Predicted_Probability', ascending=False).head(10)
print("\n🎯 Top 10 Journeys with Highest Delay Risk:")
print(top_predictions[['Predicted_Probability', 'Actual']])

# Plot Probability Distribution
plt.figure(figsize=(8, 5))
sns.histplot(y_proba, bins=25, kde=True)
plt.title("Distribution of Predicted Delay Probabilities")
plt.xlabel("Predicted Probability of Delay")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("predicted_delay_probabilities.png")
plt.show()

