"""Algoritma 4: KNN (supervised, klasifikasi berdasarkan tetangga terdekat)."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

NAMA = 'KNN'


def latih(prep, X_train, y_train):
    pipe = Pipeline([('prep', clone(prep)), ('model', KNeighborsClassifier())])
    return NAMA, pipe.fit(X_train, y_train)
