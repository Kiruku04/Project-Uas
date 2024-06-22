import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Muat model dan vectorizer
model = joblib.load('model_fake_news_detector.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Judul aplikasi
st.title('Deteksi Berita Palsu')

# Input teks dari pengguna
user_input = st.text_area('Masukkan berita di sini:')

# Prediksi dan tampilkan hasil
if st.button('Deteksi'):
    transformed_input = vectorizer.transform([user_input])
    prediction = model.predict(transformed_input)
    if prediction[0] == 1:
        st.write('Ini adalah berita palsu.')
    else:
        st.write('Ini adalah berita asli.')

# Menampilkan informasi tambahan jika diperlukan
st.info("Model dan vectorizer telah dimuat. Silakan masukkan teks berita untuk mendeteksi keasliannya.")
