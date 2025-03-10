import pandas as pd
import joblib

# Carica i dati (se il dataset è disponibile anche nell'app)
df = pd.read_excel('/path/to/your/dataset.xlsx')

# Definisci X (caratteristiche) in modo simile a come hai fatto nel notebook di addestramento
X = df.drop(columns=['Flag'])

# Carica il modello
try:
    model = joblib.load('random_forest_model.pkl')
    print("Modello caricato correttamente!")
except Exception as e:
    print(f"Errore durante il caricamento del modello: {e}")
    model = None

# Verifica se il modello è stato caricato correttamente
if model is not None:
    if hasattr(model, 'feature_importances_'):
        feature_importances = model.feature_importances_
        feature_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importances
        })
        print("Importanza delle caratteristiche:")
        print(feature_df.sort_values(by='Importance', ascending=False))
    else:
        print("Errore: il modello non ha l'attributo 'feature_importances_'")
