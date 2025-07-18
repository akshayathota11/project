# modeltraining2.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("railway_cleaned.csv")
df['Is_Delayed'] = df['Journey Status'].apply(lambda x: 1 if x == 'Delayed' else 0)

# Date/time processing
df['Date of Journey'] = pd.to_datetime(df['Date of Journey'], errors='coerce')
df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Departure Time'].dt.hour
df['Day'] = df['Date of Journey'].dt.day_name()

# Features & target
features = ['Departure Station', 'Arrival Destination', 'Ticket Class', 'Ticket Type', 'Railcard',
            'Purchase Type', 'Payment Method', 'Hour', 'Day', 'Price']
X = pd.get_dummies(df[features])
y = df['Is_Delayed']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
importances.plot(kind='barh')
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance_rf.png")
plt.show()


from sklearn.metrics import classification_report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "delay_predictor_rf.pkl")
joblib.dump(X.columns.tolist(), "model_features.pkl")
print(" Model and features saved successfully!")

