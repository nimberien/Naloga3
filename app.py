import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import datetime

# ---------------------------------------
# 1. DEL: SCRAPER (Ustvari 3 ločene CSV datoteke)
# ---------------------------------------
def scrape_all_to_csv():
    base_url = "https://web-scraping.dev"
    
    # --- Produkati (vseh 6 strani) ---
    products_list = []
    for p in range(1, 7):
        res = requests.get(f"{base_url}/products?page={p}")
        soup = BeautifulSoup(res.text, 'html.parser')
        for item in soup.select('div.row.product'):
            products_list.append({
                "Name": item.select_one('h3.mb-0').text.strip(),
                "Description": item.select_one('div.short-description').text.strip()
            })
    pd.DataFrame(products_list).to_csv("products.csv", index=False, encoding='utf-8')

    # --- Testimonials ---
    testimonials_list = []
    res = requests.get(f"{base_url}/testimonials")
    soup = BeautifulSoup(res.text, 'html.parser')
    for test in soup.select('div.testimonial'):
        testimonials_list.append({
            "Content": test.select_one('p.text').text.strip()
        })
    pd.DataFrame(testimonials_list).to_csv("testimonials.csv", index=False, encoding='utf-8')

    # --- Reviews (2023) ---
    reviews_list = []
    for p in range(1, 10): # Scrape več strani za celotno leto 2023
        res = requests.get(f"{base_url}/reviews?page={p}")
        soup = BeautifulSoup(res.text, 'html.parser')
        page_reviews = soup.select('div.review')
        if not page_reviews: break
        for rev in page_reviews:
            reviews_list.append({
                "date": rev.select_one('[data-testid="review-date"]').text.strip(),
                "text": rev.select_one('[data-testid="review-text"]').text.strip()
            })
    pd.DataFrame(reviews_list).to_csv("reviews.csv", index=False, encoding='utf-8')

# Preverimo, če datoteke obstajajo
if not os.path.exists("reviews.csv") or not os.path.exists("products.csv"):
    with st.spinner("Prvič zbiram podatke in ustvarjam CSV datoteke..."):
        scrape_all_to_csv()

# ---------------------------------------
# 2. DEL: STREAMLIT APLIKACIJA
# ---------------------------------------
st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")

st.title("📊 Brand Reputation Dashboard (2023)")

@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    testimonials = pd.read_csv("testimonials.csv")
    reviews = pd.read_csv("reviews.csv")
    return products, testimonials, reviews

products, testimonials, reviews = load_data()

section = st.sidebar.radio("Navigation", ["Products", "Testimonials", "Reviews"])

# --- PRODUKTI ---
if section == "Products":
    st.subheader("🛍️ Products")
    st.dataframe(products, use_container_width=True)

# --- TESTIMONIALS ---
elif section == "Testimonials":
    st.subheader("💬 Testimonials")
    st.dataframe(testimonials, use_container_width=True)

# --- REVIEWS ---
elif section == "Reviews":
    st.subheader("📝 Reviews & Sentiment Analysis")

    if reviews.empty:
        st.error("reviews.csv is empty.")
        st.stop()

    # Procesiranje datumov
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])
    
    # Filter samo za leto 2023
    reviews_2023 = reviews[reviews["date"].dt.year == 2023].copy()
    reviews_2023["month"] = reviews_2023["date"].dt.strftime("%B %Y")

    selected_month = st.selectbox(
        "Select month (2023)",
        sorted(reviews_2023["month"].unique(), key=lambda x: datetime.datetime.strptime(x, "%B %Y"))
    )

    filtered = reviews_2023.loc[reviews_2023["month"] == selected_month].copy()
    st.caption(f"Total reviews in {selected_month}: {len(filtered)}")

    @st.cache_resource
    def load_sentiment_model():
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    classifier = load_sentiment_model()

    with st.spinner("Running sentiment analysis..."):
        # Izvedba analize
        results = classifier(filtered["text"].astype(str).tolist())
        filtered["sentiment"] = [r["label"] for r in results]
        filtered["confidence"] = [round(r["score"], 3) for r in results]

    st.subheader("📋 Sample of Analysed Reviews")
    st.dataframe(filtered[["date", "text", "sentiment", "confidence"]], use_container_width=True)

    # Vizualizacija porazdelitve
    st.subheader("📈 Sentiment Distribution")
    sentiment_counts = filtered["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)
    st.caption(f"Average model confidence: **{filtered['confidence'].mean():.3f}**")

    # --- POLEPŠAN WORD CLOUD ---
    st.subheader("☁️ Word Cloud of Reviews")
    texts = filtered["text"].dropna().astype(str)
    
    if not texts.empty and len(" ".join(texts)) > 10:
        all_text = " ".join(texts)
        
        # Ustvarjanje Word Clouda z lepšim stilom
        wc = WordCloud(
            width=1600, 
            height=800, 
            background_color="white", 
            colormap="plasma",      # Moderna barvna paleta
            stopwords=STOPWORDS, 
            max_words=100,
            collocations=False
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)
    else:
        st.warning("Not enough text to generate Word Cloud.")
