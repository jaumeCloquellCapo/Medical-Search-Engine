import nltk
nltk.download('stopwords')
nltk.download('punkt')
from textblob import TextBlob
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import pandas as pd

from bs4 import BeautifulSoup # removing HTML tags

import logging

logger = logging.getLogger('MYAPP-API')

stemming = PorterStemmer()
wnl = WordNetLemmatizer() 
stop_words = set(stopwords.words('english'))

def remove_html_tags(text):
    clean = BeautifulSoup(text, "html.parser").get_text()
    return clean

def identify_tokens(text):
    tokens = word_tokenize(text)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def stem_list(word_list):
    stemmed_list = [stemming.stem(word) for word in word_list]
    return (stemmed_list)

def remove_stops(stemmed_words):
    meaningful_words = [w for w in stemmed_words if not w in stop_words]
    return (meaningful_words)

def cleaner(data):
    logger.info("Filter english rows")
    data_en = data[data.locale == "en-US"]
    logger.info("Removed empty descriptions")
    data_en= data_en[(data_en.medical_dictionary.notnull())]
    data_en['tokens'] = tokennize(data_en.medical_dictionary)
    return (data_en)

def tokennize(text):
    logger.info("Lower case")
    text = text.apply(str)
    
    logger.info("Remove html tags")
    text['cleaner'] = text.apply(remove_html_tags)
    #text['cleaner'] = text.apply(lambda row : BeautifulSoup(row, "html.parser").get_text()) 

    logger.info("Identify tokens")
    text['words'] = text['cleaner'].apply(identify_tokens)
    logger.info("Stemming")
    text['stemmed_words'] = text['words'].apply(stem_list)
    text['stem_meaningful'] = text['stemmed_words'].apply(remove_stops)
    return (text['stem_meaningful'])


# Use TextBlob
def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words

# Use NLTK's PorterStemmer
def stemming_tokenizer(str_input):
    words = re.sub(r"[^A-Za-z0-9\-]", " ", str_input).lower().split()
    words = [porter_stemmer.stem(word) for word in words]
    return words