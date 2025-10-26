import math

def tf(term, doc):
    '''
    @param term: word to look for
    @param doc: list of words
    @return: term frequency (relative frequency) of term
    '''
    
    return doc.count(term)/ len(doc)


def idf(term, docs):
    '''
    @param term: word to look for
    @param docs: list of docs, each doc is a list of words
    @return: inverse document frequency of term
    '''
    
    num_docs_with_term = len([doc for doc in docs if term in doc])

    return math.log(len(docs) / num_docs_with_term)


def tf_idf(term, doc, docs):
    '''
    @param term: word to look for
    @param doc: specific document to calculate tf for
    @param docs: list of docs, each doc is a list of words
    @return: tf * idf of word
    '''

    return tf(term, doc) * idf(term, docs) 

