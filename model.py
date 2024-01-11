from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pickle
import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


class SentimentRecommenderModel:

    ROOT_PATH = "pickle/"
    MODEL_NAME = "sentiment-classification-xg-boost-model.pkl"
    VECTORIZER = "tfidf-vectorizer.pkl"
    RECOMMENDER = "user_final_rating.pkl"
    CLEANED_DATA = "cleaned-data.pkl"

    def __init__(self):
        self.model = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.MODEL_NAME, 'rb'))
        self.vectorizer = pd.read_pickle(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.VECTORIZER)
        self.user_final_rating = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.RECOMMENDER, 'rb'))
        self.data = pd.read_csv("dataset/sample30.csv")
        self.cleaned_data = pickle.load(open(
            SentimentRecommenderModel.ROOT_PATH + SentimentRecommenderModel.CLEANED_DATA, 'rb'))
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))



    """function to classify the sentiment to 1/0 - positive or negative - using the trained ML model"""

    def classify_sentiment(self, review_text):
        review_text = self.preprocess_text(review_text)
        X = self.vectorizer.transform([review_text])
        y_pred = self.model.predict(X)
        return y_pred

    """function to preprocess the text before it's sent to ML model"""

    def preprocess_text(self, text):

        # cleaning the review text (lower, removing punctuation, numericals, whitespaces)
        text = text.lower().strip()
        text = re.sub("\[\s*\w*\s*\]", "", text)
        dictionary = "abc".maketrans('', '', string.punctuation)
        text = text.translate(dictionary)
        text = re.sub("\S*\d\S*", "", text)

        # remove stop-words and convert it to lemma
        text = self.lemma_text(text)
        return text

    """function to get the pos tag to derive the lemma form"""

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    """function to remove the stop words from the text"""

    def remove_stopword(self, text):
        words = [word for word in text.split() if word.isalpha()
                 and word not in self.stop_words]
        return " ".join(words)

    """function to derive the base lemma form of the text using the pos tag stemming"""

    def lemma_text(self, text):
        word_pos_tags = nltk.pos_tag(word_tokenize(
            self.remove_stopword(text)))  # Get position tags
        # Map the position tag and lemmatize the word/token
        words = [self.lemmatizer.lemmatize(tag[0], self.get_wordnet_pos(
            tag[1])) for idx, tag in enumerate(word_pos_tags)]
        return " ".join(words)
