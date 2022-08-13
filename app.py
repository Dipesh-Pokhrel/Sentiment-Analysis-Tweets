from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import re
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

def count_punctuation(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(' ')), 3) * 100

app = Flask(__name__)

data = pd.read_csv('data/sentiment.tsv', sep='\t')
data.columns = ['sentiment', 'tweets']

# Feature Engineering
data['sentiment'] = data['sentiment'].map({'pos': 1, 'neg': 0})
data['clean_tweets'] = np.vectorize(remove_pattern)(data['tweets'], "@[\w]*")
tokenized_tweet = data['clean_tweets'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    data['clean_tweets'] = tokenized_tweet
data['punctuation%'] = data['tweets'].apply(count_punctuation)
data['tweet_length'] = data['tweets'].apply(lambda x: len(x) - x.count(' '))
X = data['clean_tweets']
y = data['sentiment']
cv = CountVectorizer()
X = cv.fit_transform(X)
X = pd.DataFrame(X.toarray())

# Training Model
classifier = LogisticRegression(C=0.1, max_iter=1000)
classifier.fit(X, y)