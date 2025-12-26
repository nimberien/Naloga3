import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Nadzorna plošča ugleda blagovne znamke", layout="wide")
st.title("📊 Nadzorna plošča ugleda blagovne znamke")

# Load data
products = pd.read_csv("products.csv")
testimonials = pd.read_csv("testimonials.csv")
reviews = pd.read_csv("reviews_with_sentiment.csv")  # already contains sentiment + confidence

# Sidebar navigation
section = st.sidebar.radio("Izberi sekcijo", ["Izdelki", "Pričevanja", "Ocene"])

# IZDELKI
if section == "Izdelki":
    st.markdown("## 🛍️ Pregled izdelkov")
    st.markdown("---")
    st.dataframe(products, use_container_width=True)

# PRIČEVANJA
elif section == "Pričevanja":
    st.markdown("## 💬 Pričevanja strank")
    st.markdown("---")
    st.dataframe(testimonials, use_container_width=True)

# OCENE
elif section == "Ocene":
    st.markdown("## 📝 Analiza ocen")
    st.markdown("---")

    # Prepare date + month
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])

    # Create a Period column for proper chronological sorting
    reviews["month"] = reviews["date"].dt.to_period("M")

    # Sort months chronologically
    month_list = sorted(reviews["month"].unique())

    # Convert Period objects to readable text for display
    month_labels = [m.strftime("%B %Y") for m in month_list]

    # Select slider with month names
    selected_label = st.select_slider(
        "Izberi mesec",
        options=month_labels,
        value=month_labels[0]
    )

    # Convert selected label back to Period
    selected_period = month_list[month_labels.index(selected_label)]

    # Filter reviews
    filtered = reviews[reviews["month"] == selected_period]

    # Table with precomputed sentiment
    st.dataframe(
        filtered[["date", "text", "stars", "sentiment", "confidence"]],
        use_container_width=True
    )

    # Sentiment distribution
    st.markdown("### 📈 Porazdelitev sentimenta")
    sentiment_counts = filtered["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    avg_conf = filtered["confidence"].mean()
    st.caption(f"Povprečna zanesljivost modela za ta mesec: **{round(avg_conf, 3)}**")

    # Word Cloud
    st.markdown("### ☁️ Oblak besed iz ocen")
    all_text = " ".join(filtered["text"].astype(str))

    if len(all_text.strip()) > 10:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis"
        ).generate(all_text)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Premalo besedila za generiranje oblaka besed.")
