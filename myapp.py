import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

# Download NLTK data
nltk.download('vader_lexicon')

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

def get_synonyms(word):
    synonyms = set()

    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))

    return list(synonyms)

def paraphrase_text(text):
    tokens = word_tokenize(text)
    paraphrased_text = []

    for token in tokens:
        synonyms = get_synonyms(token)

        if synonyms:
            paraphrased_text.append(synonyms[0])
        else:
            paraphrased_text.append(token)
    return ' '.join(paraphrased_text)

# Add title and description
st.title("AI Sentiment Analysis App")
st.write("Gary Au, Sharon Hung")
st.write("Set M, Group 4")

# Create a textbox input for the user to enter text
user_input = st.text_input("Enter your text to analysis here:")

# Perform sentiment analysis when user submits input
if st.button("Analyze Sentiment"):
    if user_input:
        # Analyze sentiment
        sentiment_scores = sid.polarity_scores(user_input)

        # Get the sentiment score and label
        sentiment_score = sentiment_scores["compound"]
        sentiment_label = "Positive" if sentiment_score > 0 else "Negative"

        # Output text
        output_text = paraphrase_text(user_input)

        # Display sentiment analysis results
        st.write(f"Sentiment: {sentiment_label} Score: {abs(sentiment_score):.2f}")

        # Display the most similar sentence
        st.write(f"Alternate statement: {output_text}")
    else:
        st.warning("Please enter some text for analysis.")
