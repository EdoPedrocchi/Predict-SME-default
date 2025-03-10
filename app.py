import joblib
import pandas as pd
import streamlit as st

# Carica il modello
try:
    model = joblib.load('random_forest_model.pkl')
    st.write("Modello caricato correttamente!")
except FileNotFoundError:
    st.write("Errore: il modello non è stato trovato.")
    model = None
except Exception as e:
    st.write(f"Errore durante il caricamento del modello: {e}")
    model = None

# Verifica che il modello sia caricato correttamente e abbia l'attributo feature_importances_
if model is not None:
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        })
        st.write("Importanza delle caratteristiche:")
        st.write(feature_df.sort_values(by='Importance', ascending=False))
    else:
        st.write("Errore: il modello non ha l'attributo 'feature_importances_'")
