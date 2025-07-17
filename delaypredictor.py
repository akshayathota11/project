import pandas as pd
import joblib

# Loading training model
model = joblib.load("delay_predictor_rf.pkl")
features = joblib.load("model_features.pkl")

#  one passenger trip 
sample_input = {
    'Departure Station': 'London Kings Cross',
    'Arrival Destination': 'Edinburgh Waverley',
    'Ticket Class': 'Standard',
    'Ticket Type': 'Advance',
    'Railcard': 'No Railcard',
    'Purchase Type': 'Online',
    'Payment Method': 'Credit Card',
    'Hour': 17,
    'Day': 'Friday',
    'Price': 55.00
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

input_encoded = pd.get_dummies(input_df)

# Align columns with training features
input_encoded = input_encoded.reindex(columns=features, fill_value=0)

# Predict
prediction = model.predict(input_encoded)[0]

# Translate prediction
result = "Delayed" if prediction == 1 else "On Time"
print(f"\n Predicted Journey Status: {result}")

