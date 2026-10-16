"""Algoritma 3: SVM (supervised, mencari batas pemisah terbaik antar kelas)."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

NAMA = 'SVM'


def latih(prep, X_train, y_train):
    pipe = Pipeline([('prep', clone(prep)), ('model', SVC(class_weight='balanced', random_state=42))])
    return NAMA, pipe.fit(X_train, y_train)
