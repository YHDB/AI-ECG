
# ECG Anomaly Detection - Projet Deep Learning

## Présentation

Ce projet vise à développer une solution automatisée de détection d’anomalies dans les signaux ECG, à partir de la base de données MIT-BIH Arrhythmia Database. En utilisant un pipeline complet basé sur Python, nous avons conçu un système allant du téléchargement des données jusqu’à la visualisation des anomalies détectées.

## Structure du pipeline

- **Téléchargement automatique des enregistrements depuis PhysioNet**
- **Extraction de fenêtres centrées sur les battements annotés**
- **Prétraitement des signaux (normalisation)**
- **Entraînement d’un modèle de type MLP**
- **Évaluation du modèle et affichage de la courbe d’apprentissage**
- **Fonction de prédiction et visualisation des anomalies sur signal brut**

## Modèle utilisé

Le modèle entraîné est un réseau de neurones multicouches (MLP) avec 2 couches cachées, ReLU, Dropout et sortie sigmoïde. Il permet de classifier chaque battement comme normal ou anormal.

```python
model = Sequential([
    Dense(64, activation='relu', input_shape=(200,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

## Résultats

- **Accuracy sur test :** ~99 %
- **Recall des anomalies :** ~71 %
- **Visualisation interactive des anomalies sur les signaux ECG**
- **Matrice de confusion et rapport de classification disponibles**

## Dépendances

```bash
pip install wfdb numpy scikit-learn tensorflow matplotlib
```

## Exécution

1. Lancer le notebook ou script `ecg_project.ipynb`
2. Télécharger automatiquement les fichiers ECG
3. Modifier la ligne :
```python
predict_ecg('111', model, scaler)
```
pour tester un autre enregistrement.

## Cas d'usage

| Application           | Description                                  |
|----------------------|----------------------------------------------|
| Suivi Holter         | Détection temps réel d’anomalies à domicile |
| Hôpital connecté     | Analyse automatique d’enregistrements ECG    |
| Objets connectés     | Intégration potentielle dans des wearables   |

## Auteurs

Projet académique réalisé dans le cadre d’une étude sur les applications de l’IA en santé. Toute contribution est bienvenue.

