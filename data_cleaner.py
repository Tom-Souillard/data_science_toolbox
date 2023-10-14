import pandas as pd

def clean_data(dataframe):
    # Supprimer les lignes avec des valeurs nulles
    dataframe.dropna(inplace=True)
    # Remplacer les valeurs manquantes par la moyenne (pour les colonnes numériques)
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            dataframe[column].fillna(dataframe[column].mean(), inplace=True)
    return dataframe

# Utilisation
# df = pd.read_csv('chemin/vers/fichier.csv')
# clean_data(df)
