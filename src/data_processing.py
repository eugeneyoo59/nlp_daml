import pandas as pd
import numpy as np
from utils.tf_idf import tf_idf_vectorize
from utils.preprocessing import preprocess
from models.lstm import train_lstm_on_df
from gensim.models import Word2Vec


df = pd.read_csv("data/WELFake_Dataset.csv")
df = df.rename(columns={"Unnamed: 0": "id", "text": "tokens"})

df['tokens'] = df['tokens'].astype(str).apply(preprocess)

print('starting word2vec training')

# train word2vec on dataset

token_lists = df['tokens'].apply(lambda x: x.split()).tolist()

w2v_model = Word2Vec(
    sentences=token_lists,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

print('finished word2vec training')


w2v_model.wv.save('word2vec.wordvectors')


# train/test lstm
model, history, (X_test, y_test_cat) = train_lstm_on_df(df)


# ---------- tests ----------

# # test tf-idf vectorization
# sample = df.sample(50)
# doc = preprocess(sample.iloc[0]['tokens'])
# docs = sample['tokens'].fillna('').tolist()
# docs = [preprocess(d) for d in docs]
# print(docs[:5])
# print([item for item in tf_idf_vectorize(doc, docs).items() if item[1] != 0])

# print(preprocess(df.at[0, 'tokens']))

#print(df.info())

#print(df.head(10))

#print(df.shape())
