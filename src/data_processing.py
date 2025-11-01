import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

df = pd.read_csv("data/WELFake_Dataset.csv")
df = df.rename(columns={"Unnamed: 0": "id",  "text": "tokens"})

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def process_text(text):
    #to lower case
    text = text.lower() 

    #remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    #stopwords, stemming, lemmatization
    words = nltk.word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

#print(df.info())

#print(df.head(10))

#print(df.shape())
