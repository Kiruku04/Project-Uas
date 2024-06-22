import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import joblib

nltk.download('stopwords')
print(stopwords.words('english'))

# Memuat dataset ke dalam DataFrame pandas
news_dataset = pd.read_csv(r'D:\dokumen tugas kuliah\Semester 4\Matkul Kecerdasaan Buatan\AI\train.csv')
print(news_dataset.shape)
print(news_dataset.head())

# Menghitung jumlah nilai yang hilang dalam dataset
print(news_dataset.isnull().sum())

# Mengganti nilai null dengan string kosong
news_dataset = news_dataset.fillna('')

# Menggabungkan nama penulis dan judul berita
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Memisahkan data dan label
X = news_dataset['content'].values
Y = news_dataset['label'].values

port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

news_dataset['content'] = news_dataset['content'].apply(stemming)

# Mengubah data tekstual menjadi data numerik
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

model = LogisticRegression()
model.fit(X_train, Y_train)

# Skor akurasi pada data pelatihan
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Skor akurasi data pelatihan: ', training_data_accuracy)

# Skor akurasi pada data pengujian
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Skor akurasi data pengujian: ', test_data_accuracy)

# Simpan model dan vectorizer menggunakan joblib
joblib.dump(model, 'model_fake_news_detector.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
