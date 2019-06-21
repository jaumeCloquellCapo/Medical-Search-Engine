

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
import time

import pandas as pd

from tokenizer import cleaner, tokennize, stem_list,remove_stops, identify_tokens

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

numpy.set_printoptions(threshold=sys.maxsize)


if (__name__ == "__main__") :

    parser = argparse.ArgumentParser(description='Create a ArcHydro schema')
    parser.add_argument('--path', metavar='path', required=True, help='the path to csv file')
    parser.add_argument('--verbose', metavar='path', required=False, default = False, help='increase output verbosity')

    logging.config.fileConfig('logging.ini')

    args = parser.parse_args()
    data = pd.read_csv(args.path, sep = ";")
    
    data_en = cleaner(data)
    # data_en.to_csv("data_c.csv", sep='\t')
    
    # Split data_en on training and text
    #y = data_en.id_specialization
    #X = data_en.drop('id_specialization', axis=1)
    
    #X_train, X_test = train_test_split(data_en,test_size=0.2)

    train_documents = [TaggedDocument(data_en['tokens'][ind], str(data_en['id_specialization'][ind])) for ind in data_en.index]
    
    cores = multiprocessing.cpu_count()
    epochs = 1000

    simple_models = [
        # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, dm_concat=1, vector_size=100, hs=0, min_count=2, workers=cores,epochs=epochs),
        # PV-DBOW 
        Doc2Vec(dm=0, vector_size=100, hs=0, min_count=2, workers=cores,epochs=epochs),
        # PV-DM w/average
        Doc2Vec(dm=1, dm_mean=1, vector_size=100, hs=0, min_count=2, workers=cores,epochs=epochs),
    ]

    model = simple_models[2]
    #Essentially, the vocabulary is a dictionary (accessible via model.vocab) of all of the unique words extracted from the training corpus along with the count (e.g., model.vocab['penalty'].count for counts for the word penalty).
    model.build_vocab([x for x in train_documents])

    model.train(train_documents, total_examples=model.corpus_count, epochs=model.epochs)
    
    #Testin

    tokens = "I have a pain in my nose"

    text = identify_tokens(tokens)
    text= stem_list(text)
    text = remove_stops(text)

    print(text)

    # You can now infer a vector for any piece of text without having to re-train the model by passing a list of words to the 
    # model.infer_vector function. This vector can then be compared with other vectors via cosine similarity.

    new_vector = model.infer_vector(text ,alpha=0.001 ,steps = 5)
    #print(len(model.docvecs))
    tagsim = model.docvecs.most_similar([new_vector], topn=len(model.docvecs))
    #print(tagsim)
    
    
    for r in tagsim:
        esp = data_en[data_en['id_specialization'] == int(r[0])]
        print(r)
        print(esp['tokens'])

