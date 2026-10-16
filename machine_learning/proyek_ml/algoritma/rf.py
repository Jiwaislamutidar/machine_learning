"""Algoritma 1: Random Forest (supervised, kumpulan pohon keputusan).
Menerima data siap dari file utama, mengembalikan model yang sudah dilatih."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

NAMA = 'Random Forest'
PARAM = {
    'model__n_estimators': [200, 300, 500],
    'model__max_depth': [None, 8, 12, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2'],
}


def latih(prep, X_train, y_train, tuning=True):
    pipe = Pipeline([('prep', clone(prep)),
                     ('model', RandomForestClassifier(class_weight='balanced', random_state=42))])
    if tuning:   # cari parameter terbaik (skor F1)
        cari = RandomizedSearchCV(pipe, PARAM, n_iter=15, cv=3, scoring='f1', random_state=42, n_jobs=-1)
        cari.fit(X_train, y_train)
        print('Parameter terbaik RF:', cari.best_params_)
        return NAMA + ' (tuned)', cari.best_estimator_
    return NAMA, pipe.fit(X_train, y_train)
