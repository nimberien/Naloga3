import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Brand Reputation Dashboard", layout="wide")
st.title("📊 Brand Reputation Dashboard (precomputed sentiment)")

# Load data
products = pd.read_csv("products.csv")
testimonials = pd.read_csv("testimonials.csv")
reviews = pd.read_csv("reviews_with_sentiment.csv")  # already contains sentiment + confidence

section = st.sidebar.radio("Choose section", ["Products", "Testimonials", "Reviews"])

# PRODUCTS
if section == "Products":
    st.subheader("🛍️ Products")
    st.dataframe(products, use_container_width=True)

# TESTIMONIALS
elif section == "Testimonials":
    st.subheader("💬 Testimonials")
    st.dataframe(testimonials, use_container_width=True)

# REVIEWS
elif section == "Reviews":
    st.subheader("📝 Reviews")

    # Prepare date + month
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews = reviews.dropna(subset=["date"])
    reviews["month"] = reviews["date"].dt.strftime("%B %Y")

    selected_month = st.selectbox("Select month", sorted(reviews["month"].unique()))
    filtered = reviews[reviews["month"] == selected_month]

    st.write(f"Showing {len(filtered)} reviews from **{selected_month}**")

    # Table with precomputed sentiment
    st.dataframe(
        filtered[["date", "text", "stars", "sentiment", "confidence"]],
        use_container_width=True
    )

    # Sentiment distribution
    st.subheader("📈 Sentiment distribution")
    sentiment_counts = filtered["sentiment"].value_counts()
    st.bar_chart(sentiment_counts)

    avg_conf = filtered["confidence"].mean()
    st.caption(f"Average model confidence for this month: **{round(avg_conf, 3)}**")

    # Word Cloud
    st.subheader("☁️ Word Cloud of Reviews")
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
        st.warning("⚠️ Not enough text to generate a word cloud.")
