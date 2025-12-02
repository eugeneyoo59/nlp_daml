import pandas as pd
import numpy as np
from utils.tf_idf import tf_idf_vectorize
from utils.preprocessing import preprocess
from src.models.lstm import train_lstm_on_df

df = pd.read_csv("data/WELFake_Dataset.csv")
df = df.rename(columns={"Unnamed: 0": "id", "text": "tokens"})

df['tokens'] = df['tokens'].astype(str).apply(preprocess)

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
