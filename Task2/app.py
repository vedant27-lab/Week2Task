# streamlit_app.py

import streamlit as st
import pandas as pd
from nlp_pipeline import process_text

st.set_page_config(page_title="NLP Text Analytics", layout="wide")
st.title("📚 NLTK-Powered Text Analytics App")

# Sidebar
st.sidebar.header("Upload a Text File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt file", type=["txt"])

# Guard: Process only if file uploaded
if uploaded_file:
    try:
        text = uploaded_file.read().decode("utf-8")
        result = process_text(text)

        # Guard: check result before proceeding
        if result:
            page = st.sidebar.radio("Navigate", ["Data Explorer", "Analysis Dashboard"])

            # ------------------ PAGE 1: Data Explorer ------------------
            if page == "Data Explorer":
                st.subheader("📝 Raw Text")
                st.text_area("Original Text", text, height=200)

                st.subheader("🔤 Cleaned Tokens (First 50)")
                st.write(result["tokens"][:50])

                st.subheader("📊 Word Frequency Table")
                st.dataframe(result["frequency"])

            # ------------------ PAGE 2: Analysis Dashboard ------------------
            elif page == "Analysis Dashboard":
                st.subheader("📈 Top Bigrams")
                st.dataframe(result["bigrams"])

                st.subheader("💬 Sentiment Analysis")
                st.json(result["sentiment"])

                # Bar chart
                st.subheader("📊 Top Word Frequencies")
                top_freq = result["frequency"].head(20)
                st.bar_chart(top_freq.set_index("Word"))

                # Sentiment breakdown
                sentiment = result["sentiment"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Compound", sentiment["compound"])
                col2.metric("Positive", sentiment["pos"])
                col3.metric("Neutral", sentiment["neu"])
                col4.metric("Negative", sentiment["neg"])
        else:
            st.error("Error: Text processing failed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("👈 Please upload a `.txt` file from the sidebar.")
