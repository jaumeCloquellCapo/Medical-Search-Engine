
def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectors



def run(train_corpus):
    cores = multiprocessing.cpu_count()
    print('num of cores is %s' % cores)
    gc.collect()


    print('reading training corpus from')

    
    model = Doc2Vec(size=model_dimensions, window=10, min_count=3, sample=1e-4, negative=5, workers=cores, dm=1)
    print('building vocabulary...')
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=20)
    
    model.save(doc2vec_model)
    model.save_word2vec_format(word2vec_model)

    print('total docs learned %s' % (len(model.docvecs)))
