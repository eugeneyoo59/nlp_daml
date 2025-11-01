import pandas as pd
from utils.preprocessing import preprocess

df = pd.read_csv("data/WELFake_Dataset.csv")
df = df.rename(columns={"Unnamed: 0": "id", "text": "tokens"})

# print(preprocess(df.at[0, 'tokens']))

#print(df.info())

#print(df.head(10))

#print(df.shape())
