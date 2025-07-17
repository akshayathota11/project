# model_train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("railway_cleaned.csv")

# Create binary target variable
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Convert time and date
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Departure Time'].dt.hour
df['Day'] = df['Date of Journey'].dt.day_name()

# Select features
features = ['Departure Station', 'Arrival Destination', 'Ticket Class', 'Ticket Type', 'Railcard', 
            'Purchase Type', 'Payment Method', 'Hour', 'Day', 'Price']
X = df[features]
y = df['Is_Delayed']

# One-hot encoding for categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

