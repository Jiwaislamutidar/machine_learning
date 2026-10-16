import joblib, pandas as pd, streamlit as st

art = joblib.load('model_prediksi_belanja.joblib')
st.title('Prediksi Minat Beli Pengunjung')
st.caption(f"Model: {art['name']}")

col1, col2 = st.columns(2)
with col1:
    product = st.number_input('Jumlah halaman produk dilihat', 0, 500, 20)
    product_dur = st.number_input('Durasi halaman produk (detik)', 0.0, 20000.0, 600.0)
    page_values = st.number_input('PageValues', 0.0, 400.0, 0.0)
with col2:
    bounce = st.slider('BounceRates', 0.0, 0.2, 0.02)
    exit_ = st.slider('ExitRates', 0.0, 0.2, 0.04)
    month = st.selectbox('Bulan', art['months'])
    vtype = st.selectbox('Tipe pengunjung', art['visitor_types'])
    weekend = st.checkbox('Akhir pekan')

if st.button('Prediksi'):
    row = dict(art['defaults'])
    row.update({'ProductRelated': product, 'ProductRelated_Duration': product_dur,
                'PageValues': page_values, 'BounceRates': bounce, 'ExitRates': exit_,
                'Month': month, 'VisitorType': vtype, 'Weekend': int(weekend)})
    X_new = pd.DataFrame([row])[art['features']]
    m = art['model']
    if hasattr(m, 'predict_proba'):
        st.metric('Peluang membeli', f"{m.predict_proba(X_new)[0, 1]:.1%}")
    st.success('Cenderung MEMBELI' if m.predict(X_new)[0] == 1 else 'Cenderung TIDAK membeli')
