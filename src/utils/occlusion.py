import numpy as np

def vectorize_tokens(tokens, max_len, embedding_dim, wv, word2vec):
    '''
    vectorize one token list into shape
    '''
    X = np.zeros((1, max_len, embedding_dim), dtype=np.float32)
    j = 0
    for tok in tokens:
        if j >= max_len:
            break
        try:
            X[0, j, :] = word2vec(tok)
            j += 1
        except KeyError:
            continue
    return X


def explain_example(
        model,
        tokens,
        wv,
        word2vec,
        max_len=300,
        class_idx=0,
        top_k=20):
    '''
    explain prediction for one example by occlusion method
    remove each token and see how the probability for class_idx changes
    bigger drops = more important tokens
    '''
    embedding_dim = wv.vector_size

    # base prediction
    X_base = vectorize_tokens(tokens, max_len, embedding_dim, wv, word2vec)
    base_prob = float(model.predict(X_base)[0, class_idx])

    importances = []
    for i, tok in enumerate(tokens):
        if len(tok) == 1:
            continue # skip 1-char tokens

        tokens_occluded = tokens[:i] + tokens[i+1:]
        X_occ = vectorize_tokens(tokens_occluded, max_len, embedding_dim, wv, word2vec)
        prob_occ = float(model.predict(X_occ)[0, class_idx])
        delta = base_prob - prob_occ
        importances.append((tok, delta))

    importances.sort(key=lambda x: x[1], reverse=True)
    return base_prob, importances[:top_k]
