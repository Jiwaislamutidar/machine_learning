"""Algoritma 6: K-Means (UNSUPERVISED, mengelompokkan pengunjung tanpa memakai label Revenue)."""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

NAMA = 'K-Means'
FITUR = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
         'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']


def cari_k(df, rentang=range(2, 8)):
    """Silhouette score tiap k -> dict {k: skor}. Makin tinggi makin baik."""
    Xb = StandardScaler().fit_transform(df[FITUR])
    return {k: silhouette_score(Xb, KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xb),
                                sample_size=3000, random_state=42) for k in rentang}


def segmentasi(df, k):
    """Tambah kolom 'Cluster', kembalikan (df, profil rata-rata tiap cluster)."""
    df = df.copy()
    df['Cluster'] = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(
        StandardScaler().fit_transform(df[FITUR]))
    profil = df.groupby('Cluster')[FITUR + ['Revenue']].mean().round(2)
    profil['Jumlah'] = df['Cluster'].value_counts().sort_index()
    return df, profil.rename(columns={'Revenue': 'Rasio_Beli'})
