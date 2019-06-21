from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
import sklearn

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

import pandas as pd
import argparse
import json

from nltk.corpus import stopwords
import multiprocessing
import nltk
import csv
import logging.config
import logging
import sys
import numpy
import numpy as np

from tokenizer import cleaner, tokennize, stem_list,remove_stops, identify_tokens, remove_html_tags, textblob_tokenizer



class SearchEngine():
    def __init__(self, label_names, X_train, y_train):
        self.k = len(y_train) # K is the number of clases, in this case, specializations
        self.label_names = label_names
        self.X_train, self.y_train = X_train, y_train

    def fit(self):
        # min_df: This corresponds to the minimum number of documents that should contain this feature.
        # max_df: we should include only those words that occur in a maximum of 70% of all the documents
        self.vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=1500, min_df=5, max_df=0.4, stop_words=stopwords.words('english'))

        X_train_vect = self.vectorizer.fit_transform(self.X_train)
        self.tfidf_transformer = TfidfTransformer()
        X_train_trans = self.tfidf_transformer.fit_transform(X_train_vect)
        
        # Print TF and TFIDF
        #print(*list(X_train_vect.toarray()), sep = "\n")
        #print(*list(X_train_trans.toarray()), sep = "\n")

        # Uncomment the model to use
        #self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        #self.classifier = RandomForestClassifier(n_estimators=500, max_features=0.25, criterion="entropy", class_weight="balanced")
        self.classifier = BaggingClassifier(n_estimators =25, max_features=0.25)
        #self.classifier = GradientBoostingClassifier(n_estimators =100, learning_rate =0.1, max_depth=6, min_samples_leaf =1, max_features=1.0) clf.fit(X, training_set_y)
        #self.classifier = MultinomialNB()

        self.classifier.fit(X_train_trans, self.y_train)

    def predict(self, X_test):
        X_test_vect = self.vectorizer.transform(X_test)
        X_test_trans = self.tfidf_transformer.transform(X_test_vect)
        y_pred = self.classifier.predict(X_test_trans)
        return y_pred

    def predict_single(self, doc):
        X_test_vect = self.vectorizer.transform([doc])
        X_test_trans = self.tfidf_transformer.transform(X_test_vect)
        y_pred = zip(self.classifier.classes_, self.classifier.predict_proba(X_test_trans)[0])
        y_pred = sorted([(self.label_names[ind], score) for ind, score in y_pred], key=lambda x: -x[1])
        return y_pred

    def report(self, X_test, y_test, y_pred):
        print(classification_report(y_test, y_pred, target_names=self.label_names, digits=4))

        total = 0
        same = 0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                same += 1
            total += 1
        print(total, same)

def main(train_samples):

    label_names = sorted(train_samples.keys())

    X_train, y_train = [], []
    for label_name, docs in train_samples.items():
        label_index = label_names.index(label_name)
        for doc in docs:
            X_train.append(doc)
            y_train.append(label_index)

    model = SearchEngine(label_names, X_train, y_train)

    model.fit()

    X_test = [
        'invasive surgery',
        'cosmetic treatment'
    ]

    for doc in X_test:
        top_preds = model.predict_single(doc)[:10]
        print(doc)
        for label, score in top_preds:
            print('\t{}\t{}'.format(label, score))

    #y_pred = model.predict(X_test)


if __name__ == '__main__':
    stemmer = WordNetLemmatizer()
    parser = argparse.ArgumentParser(description='Search Engine')
    parser.add_argument('--path', metavar='path', required=True, help='the path to csv file')
    parser.add_argument('--verbose', metavar='path', required=False, default = False, help='increase output verbosity')

    logging.config.fileConfig('logging.ini')

    args = parser.parse_args()

    # Read data from cvs
    data = pd.read_csv(args.path, sep = ";")

    #Preprocess data

    #Filter only lang == EN
    data_en = data[data.locale == "en-US"]

    # Remove empty docs
    data_en= data_en[(data_en.medical_dictionary.notnull())]

    data = data_en[['id_specialization','medical_dictionary']]
    #data.medical_dictionary = data.medical_dictionary.apply(remove_html_tags)
    
    dicttionary = {}
    for index, row in data.iterrows():
        # TODO: Remove words lower 1

        # Converting to Lowercase
        document =row.medical_dictionary
        document = remove_html_tags(document)
        document = document.lower()

        # Lemmatization
        document = document.split()
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        dicttionary[row['id_specialization']] = [document]

    main(dicttionary)
