import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from transformers import pipeline

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Brand Reputation Dashboard 2023",
    layout="wide"
)

@st.cache_data
def load_data():
    products = pd.read_csv("products.csv")
    reviews = pd.read_csv("reviews.csv")
    testimonials = pd.read_csv("testimonials.csv")
    return products, reviews, testimonials

@st.cache_resource
def load_sentiment_model():
    # Lahko zamenjaš model če želiš
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

# -----------------------------------
# LOAD DATA
# -----------------------------------
products_df, reviews_df, testimonials_df = load_data()

# Poskrbimo, da je stolpec 'date' v datetime formatu
if "date" in reviews_df.columns:
    reviews_df["date"] = pd.to_datetime(reviews_df["date"], errors="coerce")

# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("Navigation")

section = st.sidebar.radio(
    "Select section:",
    ["Products", "Testimonials", "Reviews"]
)

st.sidebar.markdown("---")
st.sidebar.write("Brand Reputation Dashboard – 2023")

# -----------------------------------
# PRODUCTS SECTION
# -----------------------------------
if section == "Products":
    st.title("Products")
    st.write("Below is the list of scraped products.")
    st.dataframe(products_df, use_container_width=True)

# -----------------------------------
# TESTIMONIALS SECTION
# -----------------------------------
elif section == "Testimonials":
    st.title("Testimonials")
    st.write("Below is the list of scraped testimonials.")

    # Če želiš bolj 'list' format
    for i, row in testimonials_df.iterrows():
        st.markdown(f"- {row['text']}")

# -----------------------------------
# REVIEWS SECTION (CORE FEATURE)
# -----------------------------------
else:
    st.title("Reviews – Sentiment Analysis for 2023")

    # Filtriraj samo leto 2023
    reviews_2023 = reviews_df[
        (reviews_df["date"].dt.year == 2023)
    ].copy()

    if reviews_2023.empty:
        st.warning("No reviews found for 2023.")
        st.stop()

    # Ustvarimo mapping mesecov
    month_names = [
        "January 2023", "February 2023", "March 2023", "April 2023",
        "May 2023", "June 2023", "July 2023", "August 2023",
        "September 2023", "October 2023", "November 2023", "December 2023"
    ]
    month_numbers = list(range(1, 13))
    month_map = dict(zip(month_names, month_numbers))

    st.subheader("1. Select month")
    selected_month_label = st.select_slider(
        "Choose a month in 2023:",
        options=month_names,
        value="January 2023"
    )
    selected_month = month_map[selected_month_label]

    # Filtriraj po izbranem mesecu
    filtered = reviews_2023[reviews_2023["date"].dt.month == selected_month].copy()

    st.write(f"Number of reviews in {selected_month_label}: **{len(filtered)}**")

    if filtered.empty:
        st.warning("No reviews for this month.")
        st.stop()

    # -----------------------------------
    # SENTIMENT ANALYSIS
    # -----------------------------------
    st.subheader("2. Sentiment analysis (Hugging Face Transformers)")

    sentiment_model = load_sentiment_model()

    # Za performance: analiziraš samo text stolpec
    texts = filtered["text"].tolist()
    results = sentiment_model(texts)

    # Dodaj rezultate v dataframe
    filtered["sentiment"] = [r["label"] for r in results]
    filtered["confidence"] = [float(r["score"]) for r in results]

    # Prikaži tabelo
    st.write("Sample of analysed reviews:")
    st.dataframe(
        filtered[["date", "text", "sentiment", "confidence"]].head(20),
        use_container_width=True
    )

    # -----------------------------------
    # VISUALIZATION
    # -----------------------------------
    st.subheader("3. Sentiment distribution")

    # Štejemo positive/negative
    counts = (
        filtered.groupby("sentiment")
        .agg(
            count=("sentiment", "size"),
            avg_confidence=("confidence", "mean")
        )
        .reset_index()
    )

    # Bar chart (count)
    chart = (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("sentiment:N", title="Sentiment"),
            y=alt.Y("count:Q", title="Number of reviews"),
            color=alt.Color("sentiment:N"),
            tooltip=[
                alt.Tooltip("sentiment:N", title="Sentiment"),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("avg_confidence:Q", title="Avg confidence", format=".3f")
            ]
        )
        .properties(
            width=600,
            height=400,
            title=f"Sentiment distribution for {selected_month_label}"
        )
    )

    st.altair_chart(chart, use_container_width=True)

    # Povprečni confidence za info
    avg_conf_overall = filtered["confidence"].mean()
    st.markdown(
        f"**Average model confidence for this month:** {avg_conf_overall:.3f}"
    )
