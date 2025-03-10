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

# Se il modello è caricato correttamente, carica i dati e definisci X
if model is not None:
    try:
        # Carica il dataset (modifica il percorso con il tuo file)
        df = pd.read_excel('/Users/pedrocchiedoardo/Desktop/stramlit/SME/Dataset2_Companies.xlsx')  # Aggiungi il percorso corretto del dataset

        # Definisci X (le caratteristiche del dataset, escludendo la colonna target 'Flag')
        X = df.drop(columns=['Flag'])

        # Verifica se il modello ha l'attributo 'feature_importances_'
        if hasattr(model, 'feature_importances_'):
            # Estrai le importanze delle caratteristiche dal modello
            feature_importances = model.feature_importances_

            # Crea un DataFrame con i nomi delle caratteristiche e le rispettive importanze
            feature_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': feature_importances
            })

            # Mostra le importanze delle caratteristiche
            st.write("Importanza delle caratteristiche:")
            st.write(feature_df.sort_values(by='Importance', ascending=False))

        else:
            st.write("Errore: il modello non ha l'attributo 'feature_importances_'")

    except Exception as e:
        st.write(f"Errore durante il caricamento dei dati o la definizione di X: {e}")
