import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
df = pd.read_csv("diabetes.csv")

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---- Streamlit Web App ----
st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="")
st.title("Diabetes Risk Predictor")
st.write("Move the sliders and click Predict to check diabetes risk")

# Input sliders
pregnancies = st.slider("Number of Pregnancies", 0, 17, 3)
glucose     = st.slider("Glucose Level", 0, 200, 120)
blood_press = st.slider("Blood Pressure", 0, 122, 70)
skin_thick  = st.slider("Skin Thickness", 0, 99, 20)
insulin     = st.slider("Insulin Level", 0, 846, 79)
bmi         = st.slider("BMI", 0.0, 67.1, 25.0)
dpf         = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.47)
age         = st.slider("Age", 21, 81, 30)

# Predict button
if st.button("Predict"):
    input_data = [[pregnancies, glucose, blood_press,
                   skin_thick, insulin, bmi, dpf, age]]
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("Result: High Risk of Diabetes")
        st.write("Please consult a doctor for proper diagnosis.")
    else:
        st.success("Result: Low Risk of Diabetes")
        st.write("Keep maintaining a healthy lifestyle.")

st.write(f"Model accuracy on test data: {accuracy * 100:.2f}%")