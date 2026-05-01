import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("dataset.csv")

# Convert time
df["Departure_Time"] = pd.to_datetime(df["Departure_Time"], format="%H:%M")
df["Arrival_Time"] = pd.to_datetime(df["Arrival_Time"], format="%H:%M")

df["Journey_Duration"] = (df["Arrival_Time"] - df["Departure_Time"]).dt.total_seconds() / 3600
df["Journey_Duration"] = df["Journey_Duration"].apply(lambda x: x if x > 0 else x + 24)

# Train model
X = df[["Distance", "Stops"]]
y = df["Journey_Duration"]

model = LinearRegression()
model.fit(X, y)

# UI
st.title("🚆 Train Journey Duration Predictor")

distance = st.number_input("Enter Distance (km)", min_value=0)
stops = st.number_input("Enter Number of Stops", min_value=0)

if st.button("Predict"):
    prediction = model.predict([[distance, stops]])
    st.success(f"Predicted Journey Duration: {round(prediction[0], 2)} hours")