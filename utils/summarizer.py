# Summarization utilities
from keybert import KeyBERT
from textblob import TextBlob
import streamlit as st

# Initialize KeyBERT model (lazy loading)
_kw_model = None

def get_keybert_model():
    """
    Lazy load KeyBERT model.
    
    Returns:
        KeyBERT: Initialized KeyBERT model
    """
    global _kw_model
    if _kw_model is None:
        _kw_model = KeyBERT()
    return _kw_model

def extract_keywords(text, top_n=10):
    """
    Extract keywords from text using KeyBERT.
    
    Args:
        text: Document text
        top_n: Number of keywords to extract
    
    Returns:
        list: List of (keyword, score) tuples
    """
    try:
        model = get_keybert_model()
        
        # Extract keywords with scores
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),  # Extract 1-2 word phrases
            stop_words='english',
            top_n=top_n
        )
        
        return keywords
    except Exception as e:
        st.error(f"Error extracting keywords: {str(e)}")
        return []

def analyze_sentiment(text):
    """
    Analyze sentiment of text using TextBlob.
    
    Args:
        text: Document text
    
    Returns:
        dict: Sentiment analysis results with polarity and subjectivity
    """
    try:
        # Create TextBlob object
        blob = TextBlob(text)
        
        # Get sentiment
        sentiment = blob.sentiment
        
        # Determine sentiment label
        polarity = sentiment.polarity
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        
        return {
            'polarity': round(polarity, 3),
            'subjectivity': round(sentiment.subjectivity, 3),
            'label': label
        }
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None
