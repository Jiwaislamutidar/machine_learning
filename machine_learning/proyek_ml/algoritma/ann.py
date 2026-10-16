"""Algoritma 5: ANN / MLP (supervised, jaringan saraf tiruan 2 hidden layer)."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

NAMA = 'ANN (MLP)'


def latih(prep, X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=500, random_state=42)
    return NAMA, Pipeline([('prep', clone(prep)), ('model', model)]).fit(X_train, y_train)
