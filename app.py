import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

# 1. NASTAVITVE STRANI
st.set_page_config(page_title="Brand Reputation Monitor 2023", layout="wide")

# --- PART 3: HUGGING FACE INTEGRACIJA (Optimizirano za Render) ---
@st.cache_resource
def load_sentiment_model():
    # Uporaba "Tiny" modela, da ne presežemo 512MB RAM-a na Renderju
    # device=-1 prisili uporabo procesorja (CPU)
    return pipeline("sentiment-analysis", 
                    model="prajjwal1/bert-tiny", 
                    device=-1)

sentiment_pipeline = load_sentiment_model()

# Funkcija za nalaganje podatkov
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("analyzed_reviews_2023.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception:
        return None

df = load_data()

# --- GLAVNI NASLOV ---
st.title("🚀 Brand Reputation Monitor 2023")

# --- PART 2: SIDEBAR NAVIGATION ---
st.sidebar.title("Navigacija")
section = st.sidebar.radio("Izberi razdelek:", ("Products", "Testimonials", "Reviews"))

if df is not None:
    if section == "Products":
        st.header("📦 Products")
        st.write("Seznam izdelkov pridobljenih iz sandbox okolja.")
        st.dataframe(df[['review_text']].rename(columns={'review_text': 'Product Description'}), use_container_width=True)

    elif section == "Testimonials":
        st.header("💬 Testimonials")
        st.write("Seznam pričevanj strank.")
        st.table(df[['review_text']].head(5))

    elif section == "Reviews":
        st.header("📊 Analiza mnenj - Leto 2023")

        # PART 2: Drsnik za izbiro meseca
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        
        selected_month = st.select_slider("Izberi mesec:", options=months, value="June")
        filtered_df = df[df['month'] == selected_month]

        if filtered_df.empty:
            st.warning(f"Za mesec {selected_month} ni podatkov.")
        else:
            # --- PART 4: VIZUALIZACIJA (Bar Chart) ---
            st.subheader(f"Statistika za {selected_month}")

            # Priprava podatkov za stolpčni grafikon
            chart_data = filtered_df.groupby('sentiment').agg(
                Count=('sentiment', 'count'),
                Avg_Confidence=('confidence', 'mean')
            ).reset_index()

            # ZAHTEVA: Bar Chart s Confidence Score v tooltipu
            fig = px.bar(
                chart_data, 
                x='sentiment', 
                y='Count',
                color='sentiment',
                color_discrete_map={'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c'},
                title=f"Število mnenj v mesecu {selected_month}",
                hover_data={'Avg_Confidence': ':.4f'} 
            )
            st.plotly_chart(fig, use_container_width=True)

            # Prikaz povprečne stopnje zaupanja (Advanced Part 4)
            avg_score = filtered_df['confidence'].mean()
            st.metric("Povprečna stopnja zaupanja modela", f"{avg_score:.2%}")

            # PART 3: Prikaz tabele z AI analizo
            st.write("### Podrobna tabela mnenj")
            st.dataframe(filtered_df[['date', 'review_text', 'sentiment', 'confidence']], use_container_width=True)
else:
    st.error("Manjka datoteka 'analyzed_reviews_2023.csv'!")
