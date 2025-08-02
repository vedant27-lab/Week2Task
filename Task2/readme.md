# Task 2: NLTK-Powered Text Analytics Web App (Streamlit + Pandas)

This project implements an **interactive web application** to explore and analyze text files using **NLTK**, **pandas**, and **Streamlit**.

Users can upload `.txt` files and visualize insights like word frequency, bigrams, and sentiment scores using simple point-and-click UI.

---

## 🚀 Features

✅ Upload your own `.txt` file  
✅ Text cleaning, tokenization, lemmatization, stopword removal  
✅ Frequency distribution of top words  
✅ Bigrams (collocations) detection  
✅ Sentiment analysis using VADER  
✅ Visualization with bar charts and metrics  
✅ Two-page UI:
- 📄 **Data Explorer**
- 📊 **Analysis Dashboard**

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `streamlit_app.py` | Main app interface using Streamlit |
| `nlp_pipeline.py`  | Core logic: preprocessing, NLP tasks |
| `example.txt`      | Sample text file for testing |
| `README.md`        | This documentation file |

---

## ⚙️ How to Run Locally

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Task2

2. Install Required Packages
pip install streamlit nltk pandas

3. Download NLTK Resources (First time only)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
Or add this to the top of nlp_pipeline.py (already done).

4. Run the App
streamlit run app.py

Example Text File
You can use the provided example.txt or upload any .txt file — e.g.,:

1) Tweets
2) Movie reviews
3) News articles
4) Abstracts or blogs

Screenshots

Data Explorer

1) View raw text
2) See tokens and frequency table

Analysis Dashboard

1) Top bigrams
2) Sentiment scores
3) Frequency chart + metrics

References
* https://www.nltk.org/
* https://docs.streamlit.io/
* https://github.com/cjhutto/vaderSentiment