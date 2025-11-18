import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import urllib.parse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
try:
    stop_words = set(stopwords.words('indonesian'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('indonesian'))

# ========== STREAMLIT SETUP ==========
st.set_page_config(layout="wide")
st.title("📊 Analisis Sentimen Portal Berita Detik dan Kompas")

# ========== UTILITIES ==========
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

def get_top_tfidf_words(texts, top_n=20):
    tfidf = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = tfidf.fit_transform(texts)
    scores = tfidf_matrix.sum(axis=0).A1
    words = tfidf.get_feature_names_out()
    return sorted(zip(words, scores), key=lambda x: x[1], reverse=True)

# ========== SCRAPING ==========
def get_soup(url):
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        r.raise_for_status()
        return BeautifulSoup(r.text, 'html.parser')
    except Exception:
        return None

def extract_kompas_article(link):
    soup = get_soup(link)
    if not soup:
        return None
    title_tag = soup.select_one('h1.read__title')
    paragraphs = soup.select('div.read__content p')
    title = title_tag.get_text(strip=True) if title_tag else "No Title"
    content = ' '.join(p.get_text(strip=True) for p in paragraphs)
    return {'title': title, 'link': link, 'content': content, 'source': 'Kompas'}

def extract_detik_article(link, fallback_title):
    soup = get_soup(link)
    if not soup:
        return None
    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else fallback_title
    paragraphs = soup.select('div.detail__body-text.itp_bodycontent p')
    content = ' '.join(p.get_text(strip=True) for p in paragraphs)
    return {'title': title, 'link': link, 'content': content, 'source': 'Detik'}

def scrape_kompas(keyword, limit=50):
    results = []
    count = 0
    page = 1
    while count < limit:
        url = f"https://www.kompas.com/tag/{urllib.parse.quote(keyword)}?page={page}"
        soup = get_soup(url)
        if not soup:
            break
        articles = soup.select('div.articleItem')
        for art in articles:
            if count >= limit:
                break
            try:
                a = art.select_one('a.article-link')
                if not a:
                    continue
                link = a['href']
                result = extract_kompas_article(link)
                if result:
                    results.append(result)
                    count += 1
            except Exception:
                continue
        page += 1
    return results

def scrape_detik(keyword, limit=50):
    results = []
    count = 0
    page = 1
    while count < limit:
        url = f"https://www.detik.com/search/searchall?query={urllib.parse.quote(keyword)}&sortby=time&page={page}"
        soup = get_soup(url)
        if not soup:
            break
        articles = soup.select('article')
        for art in articles:
            if count >= limit:
                break
            try:
                a = art.select_one('a')
                if not a:
                    continue
                link = a['href']
                fallback_title = art.get_text(strip=True)
                result = extract_detik_article(link, fallback_title)
                if result:
                    results.append(result)
                    count += 1
            except Exception:
                continue
        page += 1
    return results

def scrape_news(keyword, kompas_limit=200, detik_limit=200):
    kompas_results = scrape_kompas(keyword, kompas_limit)
    detik_results = scrape_detik(keyword, detik_limit)
    return pd.DataFrame(kompas_results + detik_results)

# ========== MODEL LOADING (cached) ==========
@st.cache_resource
def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

# ========== UI: Input ==========
keyword = st.text_input("🔎 Masukkan keyword berita:")
batch_size = 48
max_items = st.number_input("Max artikel (total dari Kompas+Detik)", min_value=10, max_value=1000, value=200, step=10)

submit_button = st.button("🚀 Proses Analisis")

if submit_button:
    if not keyword.strip():
        st.warning("Mohon masukkan keyword terlebih dahulu.")
    else:
        st.info("🔄 Sedang mencari dan menyiapkan analisis...")

        # scrape
        df = scrape_news(keyword, kompas_limit=max_items//2, detik_limit=max_items//2)

        if df.empty:
            st.warning("Tidak menemukan artikel dengan keyword tersebut.")
        else:
            # preprocessing
            df = df.drop_duplicates(subset='link').dropna(subset=['content']).reset_index(drop=True)
            df['cleaned_content'] = df['content'].apply(clean_text)

            # load model & tokenizer
            model_path = "desimdama/indonesian_roberta_news_sentiment"
            with st.spinner("Memuat model dan tokenizer..."):
                tokenizer, model = load_model_and_tokenizer(model_path)

            # device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            # prepare for batched inference
            texts = df['cleaned_content'].tolist()
            n = len(texts)
            n_batches = (n + batch_size - 1) // batch_size

            # UI elements for progress
            progress_text = st.empty()
            progress_bar = st.progress(0.0)
            start_time = time.time()

            label_map = ['Negatif', 'Netral', 'Positif']
            all_labels = []

            for b in range(n_batches):
                i0 = b * batch_size
                i1 = min(i0 + batch_size, n)
                batch_texts = texts[i0:i1]

                encoded = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
                encoded = {k: v.to(device) for k, v in encoded.items()}

                with torch.no_grad():
                    outputs = model(**encoded)

                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                labels = torch.argmax(probs, dim=1).cpu().tolist()
                all_labels.extend([label_map[l] for l in labels])

                # update progress
                elapsed = time.time() - start_time
                batches_done = b + 1
                avg_batch_time = elapsed / batches_done
                est_total = avg_batch_time * n_batches
                est_remaining = max(0.0, est_total - elapsed)

                progress_bar.progress(min(1.0, batches_done / n_batches))
                progress_text.text(
                    f"⏳ Batch {batches_done}/{n_batches} — memproses artikel {i0+1}-{i1} dari {n} | "
                    f"Estimasi selesai: {est_remaining:.1f} s"
                )

            df['sentiment'] = all_labels

            total_time = time.time() - start_time
            st.success(f"✅ Selesai menganalisis {len(df)} berita dalam {total_time:.2f} detik")

            # ========== TF-IDF ==========
            top_words = get_top_tfidf_words(df['cleaned_content'], top_n=20)
            top_df = pd.DataFrame(top_words, columns=['Kata', 'Skor TF-IDF'])

            def get_top_keywords(text, tfidf_vectorizer, top_n=5):
                tfidf_matrix = tfidf_vectorizer.transform([text])
                scores = tfidf_matrix.toarray()[0]
                indices = scores.argsort()[-top_n:][::-1]
                feature_names = tfidf_vectorizer.get_feature_names_out()
                return [feature_names[i] for i in indices if scores[i] > 0]

            tfidf_vectorizer = TfidfVectorizer(max_features=20)
            tfidf_vectorizer.fit(df['cleaned_content'])
            df['top_keywords'] = df['cleaned_content'].apply(lambda x: get_top_keywords(x, tfidf_vectorizer))

            # ========== DASHBOARD ==========
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📄 Data Berita Digital")
                df_display = df[['title', 'sentiment', 'source', 'link', 'top_keywords']].copy()
                df_display.columns = ['📰 Title', '📊 Sentiment', '🗞️ Source', '🔗 Link', '🧠 Top Keywords']
                st.dataframe(df_display)

                st.subheader("📈 Kata Penting (TF-IDF)")
                st.dataframe(top_df)

            with col2:
                st.subheader("📊 Distribusi Sentimen")
                fig1, ax1 = plt.subplots()
                sns.countplot(data=df, x='sentiment', hue='sentiment', palette='Set2', ax=ax1, dodge=False)
                legend = ax1.get_legend()
                if legend:
                    legend.remove()  
                st.pyplot(fig1)

                st.subheader("🌥️ Word Cloud")
                all_text = ' '.join(df['cleaned_content'])
                wordcloud = WordCloud(stopwords=STOPWORDS.union(stop_words), background_color='white').generate(all_text)
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                ax2.imshow(wordcloud, interpolation='bilinear')
                ax2.axis('off')
                st.pyplot(fig2)

            # ========== BERITA PER SENTIMEN ==========
            st.subheader("📰 Sampel Representatif Berdasarkan Kategori Sentimen")
            for sent in ['Positif', 'Netral', 'Negatif']:
                st.markdown(f"### {sent.capitalize()}")
                sample = df[df['sentiment'] == sent].head(1)
                if not sample.empty:
                    st.write(f"**{sample.iloc[0]['title']}**")
                    st.write(sample.iloc[0]['content'][:500] + "...")
                    st.write(f"🔗 [Link ke berita]({sample.iloc[0]['link']})")
                    st.markdown(f"**Kata kunci utama:** {', '.join(sample.iloc[0]['top_keywords'])}")
                else:
                    st.write("Tidak ada contoh berita dengan sentimen ini.")