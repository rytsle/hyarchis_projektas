# a) Prognozuoti ‚sentiment‘ pagal ‚text‘
# https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset

import pandas as pd
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

file_path = 'Tweets.csv'
df = pd.read_csv(file_path, encoding='utf-8')
df = df[['text', 'sentiment']]
df.dropna(inplace=True)
df.head()
df['text'] = df['text'].apply(lambda x: x.lower())
df['text'] = df['text'].str.replace(r'http\S+|www\S+|https\S+', '', case=False)
df['text'] = df['text'].str.replace(r'@\\w+', '', case=False)
df['text'] = df['text'].str.replace(r'#\\w+', '', case=False)
df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))


stop_words = set(stopwords.words('english'))

keep_words = ['not', 'no', 'never', 'neither', 'nor', 'none', 'hardly', 'barely', 'scarcely', "don't", 
                  "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't", "shouldn't", "isn't", "aren't"]

for word in keep_words:
    if word in stop_words:
        stop_words.remove(word)


df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
