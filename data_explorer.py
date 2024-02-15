import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    """
    Charge un fichier de données dans un DataFrame pandas.

    Args:
        filepath (str): Le chemin du fichier à charger (CSV, Excel, etc.).

    Returns:
        DataFrame: Un DataFrame pandas contenant les données chargées.
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        else:
            print("Format de fichier non supporté. Veuillez utiliser un fichier CSV ou Excel.")
            return None
        print(f"Données chargées avec succès depuis {filepath}")
        return df
    except Exception as e:
        print(f"Erreur lors du chargement des données : {e}")
        return None


def clean_data(df):
    """
    Nettoie le DataFrame en remplaçant les valeurs manquantes et en supprimant les colonnes inutiles.

    Args:
        df (DataFrame): Le DataFrame à nettoyer.

    Returns:
        DataFrame: Le DataFrame nettoyé.
    """
    # Remplacer les valeurs NaN par des valeurs spécifiques ou les supprimer
    df.fillna(value=0, inplace=True)  # Exemple simple, à adapter selon les besoins

    # Supprimer les colonnes inutiles, exemple :
    # df.drop(['ColonneInutile'], axis=1, inplace=True)

    return df


def analyze_data(df):
    """
    Effectue une analyse descriptive de base sur le DataFrame.

    Args:
        df (DataFrame): Le DataFrame à analyser.

    Returns:
        None
    """
    print("Description des données :")
    print(df.describe())

    # Ajoutez ici d'autres analyses, comme la corrélation entre colonnes :
    # print(df.corr())


def visualize_data(df):
    """
    Crée des visualisations de base pour le DataFrame.

    Args:
        df (DataFrame): Le DataFrame à visualiser.

    Returns:
        None
    """
    # Histogramme des variables numériques
    df.hist(figsize=(10, 8))
    plt.tight_layout()
    plt.show()

    # Boîte à moustaches pour visualiser les distributions
    # Remplacer 'ColonneNumerique' par le nom d'une colonne numérique spécifique
    # sns.boxplot(x=df['ColonneNumerique'])
    # plt.show()

    # Matrice de corrélation
    # sns.heatmap(df.corr(), annot=True, fmt=".2f")
    # plt.show()


if __name__ == "__main__":
    filepath = 'chemin/vers/votre/fichier.csv'  # Modifiez le chemin selon votre fichier
    df = load_data(filepath)
    if df is not None:
        df = clean_data(df)
        analyze_data(df)
        visualize_data(df)
