# Importation des bibliothèques nécessaires
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_dataset(filepath):
    """
    Charge les données depuis un fichier CSV dans un DataFrame pandas.

    Args:
        filepath (str): Chemin d'accès au fichier CSV.

    Returns:
        DataFrame: Un DataFrame contenant les données chargées.
    """
    return pd.read_csv(filepath)


def prepare_data(df, target_column):
    """
    Prépare les données pour la régression linéaire multiple, en séparant les variables indépendantes et la variable dépendante.

    Args:
        df (DataFrame): Le DataFrame contenant les données.
        target_column (str): Le nom de la colonne qui est la variable dépendante.

    Returns:
        X (DataFrame): Un DataFrame des variables indépendantes.
        y (Series): Une Series de la variable dépendante.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test.

    Args:
        X (DataFrame): Le DataFrame des variables indépendantes.
        y (Series): La Series de la variable dépendante.
        test_size (float): La proportion de l'ensemble de test.
        random_state (int): Graine pour la génération de nombres aléatoires.

    Returns:
        X_train, X_test, y_train, y_test: Les ensembles d'entraînement et de test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train):
    """
    Entraîne le modèle de régression linéaire multiple sur les données d'entraînement.

    Args:
        X_train (DataFrame): Le DataFrame des variables indépendantes d'entraînement.
        y_train (Series): La Series de la variable dépendante d'entraînement.

    Returns:
        model: Le modèle entraîné.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test et imprime les métriques de performance.

    Args:
        model: Le modèle entraîné.
        X_test (DataFrame): Le DataFrame des variables indépendantes de test.
        y_test (Series): La Series de la variable dépendante de test.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")


if __name__ == "__main__":
    # Exemple d'utilisation
    filepath = 'chemin/vers/le/fichier.csv'  # À remplacer par le chemin d'accès réel
    target_column = 'nom_colonne_cible'  # À remplacer par le nom de la colonne cible réelle
    df = load_dataset(filepath)
    X, y = prepare_data(df, target_column)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
