import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))

# Flower names
species = ["Setosa", "Versicolor", "Virginica"]

# Page title
st.title("🌸 Iris Flower Classification App")

st.write("Enter flower measurements below:")

# Input fields
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    prediction = model.predict(input_data)
    
    st.success(f"Predicted Flower: {species[prediction[0]]}")