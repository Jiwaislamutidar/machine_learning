"""Algoritma 2: XGBoost (supervised, boosting: pohon dibangun bertahap memperbaiki kesalahan)."""
from sklearn.base import clone
from sklearn.pipeline import Pipeline

NAMA = 'XGBoost'


def latih(prep, X_train, y_train):
    try:
        from xgboost import XGBClassifier
        rasio = (y_train == 0).sum() / (y_train == 1).sum()     # penyeimbang kelas
        model, nama = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.8, scale_pos_weight=rasio,
                                    eval_metric='logloss', random_state=42, n_jobs=-1), NAMA
    except ImportError:   # cadangan jika xgboost belum terpasang (pip install xgboost)
        from sklearn.ensemble import HistGradientBoostingClassifier
        model, nama = HistGradientBoostingClassifier(class_weight='balanced', random_state=42), 'HistGradientBoosting'
    return nama, Pipeline([('prep', clone(prep)), ('model', model)]).fit(X_train, y_train)
