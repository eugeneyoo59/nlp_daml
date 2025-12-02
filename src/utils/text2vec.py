import numpy as np
from gensim.test.utils import common_texts
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# TODO: replace with actual model loading (common_texts is for testing)

# load word2vec model

# # common texts for testing
# wv_model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
# word_vectors = wv_model.wv
# word_vectors.save('word2vec.wordvectors')
# wv = KeyedVectors.load('word2vec.wordvectors', mmap='r')

wv = KeyedVectors.load('word2vec.wordvectors', mmap='r')

# load doc2vec model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
dv_model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)

# define vectorizers
word2vec = lambda word: wv[word]
doc2vec = lambda doc: dv_model.infer_vector(doc)

def vectorize(doc, vectorizer=word2vec):
    '''
    @param doc: preprocessed text of words
    @param vectorizer: word2vec or doc2vec
    @return: sum of vectors of each word in doc
    '''

    if isinstance(doc, str):
        doc = doc.split()

    if vectorizer is doc2vec:
        return vectorizer(doc)
    
    vectors = []
    for token in doc:
        try:
            vectors.append(np.array(vectorizer(token)))
        except KeyError:
            continue

    if not vectors:
        return np.zeros(wv.vector_size, dtype=float)
    
    return np.sum(vectors, axis=0)

# ---------- tests ----------

print(word2vec('computer'))
print(vectorize('human computer system'))
print(vectorize('human computer system', vectorizer=doc2vec))