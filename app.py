import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Carica il modello salvato
model = joblib.load('random_forest_model.pkl')

# Funzione per fare predizioni
def make_prediction(input_data):
    prediction = model.predict(input_data)
    return prediction

# Interfaccia utente con Streamlit
st.title("Predizione con Random Forest")

# Chiedi all'utente di inserire i dati (fai attenzione ai tipi di input e alla forma dei dati)
input_data = []
for column in model.feature_importances_.index:
    value = st.number_input(f"Inserisci il valore per {column}")
    input_data.append(value)

input_data = np.array(input_data).reshape(1, -1)

if st.button("Predici"):
    prediction = make_prediction(input_data)
    st.write(f"La previsione del modello è: {prediction[0]}")
