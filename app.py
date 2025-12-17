import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# Nastavitve strani
st.set_page_config(page_title="Brand Reputation Monitor 2023", layout="wide")

# --- PART 3: HUGGING FACE INTEGRACIJA ---
# Naložimo model za globoko učenje (Transformers)
@st.cache_resource
def load_sentiment_model():
    # Uporaba zahtevanega modela distilbert za klasifikacijo Positive/Negative
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_pipeline = load_sentiment_model()

# Funkcija za nalaganje podatkov iz CSV
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("analyzed_reviews_2023.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return None

df = load_data()

st.title("🚀 Brand Reputation Monitor 2023")
st.markdown("Aplikacija za spremljanje ugleda blagovne znamke s pomočjo strojnega učenja.")

# --- PART 2: SIDEBAR NAVIGATION ---
st.sidebar.title("Navigacija")
section = st.sidebar.radio(
    "Izberi razdelek:",
    ("Products", "Testimonials", "Reviews")
)

if df is not None:
    # Razdelka Products in Testimonials (Preprost prikaz)
    if section == "Products":
        st.header("📦 Products")
        st.write("Seznam izdelkov, pridobljenih s strganjem podatkov.")
        st.dataframe(df[['review_text']].rename(columns={'review_text': 'Product Description'}), width='stretch')

    elif section == "Testimonials":
        st.header("💬 Testimonials")
        st.write("Pričevanja strank o naših storitvah.")
        st.table(df[['review_text']].head(5))

    # --- PART 2 & 3: REVIEWS (CORE FEATURE) ---
    elif section == "Reviews":
        st.header("📊 Analiza mnenj za leto 2023")

        # Seznam mesecev za drsnik (PART 2)
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        
        # ZAHTEVA: st.select_slider za izbiro meseca
        selected_month = st.select_slider(
            "Izberi mesec v letu 2023:",
            options=months,
            value="June"
        )

        # ZAHTEVA: Filtriranje podatkov na podlagi meseca
        filtered_df = df[df['month'] == selected_month]

        if filtered_df.empty:
            st.warning(f"Za mesec {selected_month} ni podatkov v 2023.")
        else:
            st.subheader(f"Rezultati za mesec: {selected_month}")
            
            # Prikaz ključnih metrik
            col1, col2 = st.columns(2)
            pos_reviews = filtered_df[filtered_df['sentiment'] == 'POSITIVE']
            col1.metric("Skupno mnenj", len(filtered_df))
            col2.metric("Pozitivna mnenja", len(pos_reviews))

            # ZAHTEVA: Prikaz mnenj s sentimentom (PART 3)
            st.write("### Podrobna AI analiza")
            st.dataframe(filtered_df[['date', 'review_text', 'sentiment', 'confidence']], width='stretch')
            
            # Grafikon za boljšo vizualizacijo
            fig = px.pie(filtered_df, names='sentiment', color='sentiment',
                         color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'},
                         title=f"Sentiment v mesecu {selected_month}")
            st.plotly_chart(fig)
else:
    st.error("Podatki niso na voljo. Najprej zaženi skripto za strganje podatkov (Naloga3.py).")