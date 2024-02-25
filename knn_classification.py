# Importation des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


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
    Prépare les données pour la classification, en séparant les variables indépendantes et la variable dépendante.

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


def split_and_train(X, y, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement et de test, entraîne le modèle k-NN.

    Args:
        X (DataFrame): Le DataFrame des variables indépendantes.
        y (Series): La Series de la variable dépendante.
        n_neighbors (int): Le nombre de voisins à utiliser.
        test_size (float): La proportion de l'ensemble de test.
        random_state (int): Graine pour la génération de nombres aléatoires.

    Returns:
        model: Le modèle k-NN entraîné.
        X_train, X_test, y_train, y_test: Les ensembles d'entraînement et de test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur l'ensemble de test et imprime le rapport de classification et la matrice de confusion.

    Args:
        model: Le modèle k-NN entraîné.
        X_test (DataFrame): Le DataFrame des variables indépendantes de test.
        y_test (Series): La Series de la variable dépendante de test.
    """
    predictions = model.predict(X_test)
    print("Rapport de classification :\n", classification_report(y_test, predictions))
    print("Matrice de confusion :\n", confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    filepath = 'chemin/vers/votre/fichier.csv'  # Remplacez par le chemin réel
    target_column = 'nom_de_la_colonne_cible'  # Remplacez par le nom réel de la colonne cible
    df = load_dataset(filepath)
    X, y = prepare_data(df, target_column)
    model, X_train, X_test, y_train, y_test = split_and_train(X, y)
    evaluate_model(model, X_test, y_test)
