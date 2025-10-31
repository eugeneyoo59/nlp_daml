import math

def tf(term, doc):
    '''
    @param term: word to look for
    @param doc: list of words
    @return: term frequency (relative frequency) of term
    '''

    if len(doc) == 0:
        raise ValueError('cannot divide by zero')
    
    return doc.count(term) / len(doc)


def idf(term, docs):
    '''
    @param term: word to look for
    @param docs: list of docs, each doc is a list of words
    @return: inverse document frequency of term
    '''
    
    num_docs_with_term = len([doc for doc in docs if term in doc])

    if num_docs_with_term == 0:
        raise ValueError('cannot divide by zero')

    if len(docs) == 0:
        raise ValueError('cannot take log of zero')

    return math.log(len(docs) / num_docs_with_term)


def tf_idf(term, doc, docs):
    '''
    @param term: word to look for
    @param doc: specific document to calculate tf for
    @param docs: list of docs, each doc is a list of words
    @return: tf * idf of word
    '''

    return tf(term, doc) * idf(term, docs)


def vectorize(doc, docs):
    '''
    @param doc: specific doc to vectorize, a list of words
    @param docs: list of docs, each doc is a list of words
    @return: dictionary mapping each unique word in docs to its tf-idf value in doc
    '''

    unique_terms = sorted(set(term for d in docs for term in d))

    return {term: tf_idf(term, doc, docs) for term in unique_terms}
