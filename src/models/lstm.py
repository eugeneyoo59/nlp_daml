import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.utils import to_categorical

from utils.text2vec import wv, word2vec

def df_to_w2v_sequences(df: pd.DataFrame, max_len: int = 300):
    '''
    convert df['tokens'] into 3d numpy array of shape (n_samples, max_len, embedding_dim)
    '''

    embedding_dim = wv.vector_size
    n_samples = len(df)

    X = np.zeros((n_samples, max_len, embedding_dim), dtype=np.float32)

    for i, tokens in enumerate(df['tokens']):
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        j = 0
        for t in tokens:
            if j >= max_len:
                break
            try:
                X[i, j, :] = word2vec(t)
                j += 1
            except KeyError:
                continue
    
    return X


def evaluate_model(model, X_test, y_test_cat):
    '''
    compute core metrics and return predictions + probabilities
    '''
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    print('classification report:')
    print(classification_report(y_true, y_pred, digits=3))

    print('confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

    return y_true, y_pred, y_pred_proba


def train_lstm_on_df(
        df: pd.DataFrame,
        max_len: int=300,
        test_size: float=0.2,
        batch_size: int=64,
        epochs: int=5,
        random_state: int=42
):
    '''
    train lstm on dataframe with
    - df['tokens']: text/token list
    - df['label']: 0/1 (fake/real)
    '''

    if 'tokens' not in df.columns:
        raise ValueError('df must contain tokens column')
    if 'label' not in df.columns:
        raise ValueError('df must contain label column')

    X = df_to_w2v_sequences(df, max_len=max_len)
    y = df['label'].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    y_train_cat = to_categorical(y_train, num_classes=2)
    y_test_cat = to_categorical(y_test, num_classes=2)

    embedding_dim = wv.vector_size

    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(max_len, embedding_dim)))
    model.add(LSTM(units=128, input_shape=(max_len, embedding_dim)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    
    history = model.fit(
        X_train,
        y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f'test loss: {test_loss:.4f}, test accuracy: {test_acc:.4f}')

    y_true, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test_cat)

    return model, history, (X_test, y_test_cat, y_true, y_pred, y_pred_proba)