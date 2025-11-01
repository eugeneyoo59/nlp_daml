import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")

def word2vec(word):
    '''
    @param word: word to vectorize
    @return: word2vec vector of word
    '''

    model = Word2Vec.load("word2vec.model")
    return model.wv[word]

def doc2vec(doc, vectorizer=word2vec):
    '''
    @param doc: processed text of words
    @param vectorizer: word2vec, sentence2vec, or doc2vec
    @return: sum of vectors of each word in doc
    '''

    doc = doc.split()
    vectors = []

    for word in doc:
        try:
            vectors.append(np.array(vectorizer(word)))
        except KeyError:
            continue

    return sum(vectors)