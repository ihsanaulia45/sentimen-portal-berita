Judul Proyek: Aplikasi Analisis Teks & Sentimen Berbasis Streamlit
Deskripsi
Aplikasi ini adalah tool berbasis web yang dibangun menggunakan Python. Aplikasi ini memiliki kemampuan untuk mengambil data teks (scraping), memproses bahasa alami (NLP), melakukan klasifikasi teks (menggunakan Machine Learning/Deep Learning), dan memvisualisasikan hasilnya.

Prasyarat (Dependencies)
Berikut adalah daftar pustaka (library) Python yang digunakan dalam proyek ini beserta fungsinya:

1. Antarmuka & Web Framework:

streamlit: Digunakan untuk membuat antarmuka web (dashboard) yang interaktif dengan mudah tanpa perlu front-end coding yang rumit.

2. Pengambilan Data (Scraping & Request):

requests: Untuk mengirim permintaan HTTP ke website target.

bs4 (BeautifulSoup): Untuk melakukan parsing HTML dan mengekstrak data teks dari halaman web.

urllib.parse: Untuk memecah dan menyusun URL (biasanya untuk encoding parameter query).

3. Pengolahan Data:

pandas: Untuk manipulasi data dalam bentuk tabel (DataFrame), seperti menyimpan hasil scraping dan preprocessing.

time: Untuk mengatur jeda waktu (sleep) saat scraping agar tidak membebani server target atau terdeteksi sebagai bot spam.

4. Natural Language Processing (NLP) & Machine Learning:

torch (PyTorch): Framework utama untuk menjalankan model Deep Learning.

transformers (Hugging Face): Untuk memuat model bahasa yang sudah dilatih (Pre-trained Models) seperti BERT/RoBERTa untuk klasifikasi teks/sentimen.

sklearn (Scikit-learn):

TfidfVectorizer: Untuk mengubah teks menjadi angka (vektor) berdasarkan bobot kata (TF-IDF).

nltk:

stopwords: Untuk menghapus kata-kata umum yang tidak bermakna (seperti "yang", "dan", "di") dalam bahasa Indonesia.

re: Untuk membersihkan teks menggunakan Regular Expressions (menghapus simbol, angka, atau pola tertentu).

5. Visualisasi Data:

matplotlib.pyplot: Library dasar untuk membuat grafik.

seaborn: Library visualisasi di atas Matplotlib untuk membuat grafik statistik yang lebih menarik.

wordcloud: Untuk membuat visualisasi "awan kata" dari kata-kata yang paling sering muncul dalam teks.