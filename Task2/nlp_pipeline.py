import nltk 
import pandas as pd
import string 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize(text: str) -> list:
    return word_tokenize(text)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def remove_stopwords(tokens: list) -> list:
    return [word for word in tokens if word not in stop_words and word.isalpha()]

def lemmatize(tokens: list)->list:
    return [lemmatizer.lemmatize(word) for word in tokens]

def word_frequency(tokens: list) -> pd.DataFrame:
    freq_dist = nltk.FreqDist(tokens)
    return pd.DataFrame(freq_dist.most_common(), columns=["Word", "Frequency"])

def get_top_bigrams(tokens: list, top_n: int = 10) -> pd.DataFrame:
    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.likelihood_ratio, top_n)
    return pd.DataFrame(bigrams, columns=["Word 1", "Word 2"])

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    return sia.polarity_scores(text)

def process_text(text: str) -> dict:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    filtered = remove_stopwords(tokens)
    lemmas = lemmatize(filtered)

    freq = word_frequency(lemmas)
    bigrams = get_top_bigrams(lemmas)
    sentiment = analyze_sentiment(text)

    return {
        "tokens": lemmas,
        "frequency": freq,
        "bigrams": bigrams,
        "sentiments": sentiment
    }

