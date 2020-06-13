import csv
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from connect_mongo import read_mongo

# Read Twitter data
# description: bio
# original author: username
# screen_name: full user name
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1, 'text': 1})[50000:100000]
# data = data.drop_duplicates(subset='original author', keep="first").reset_index()
# data = data.iloc[:50000]
# data = data.iloc[:10000]
# data = data.iloc[:100]

print(data)

pd.set_option('display.max_columns', None)
# print(data.head())
nested_data = json_normalize(data['user'])

author_df = (data['original author'])
desc = (nested_data['description'].tolist())
data['description'] = desc
text_df = (data['text'])

data = pd.concat([author_df, data['description'], text_df], axis=1, sort=False)
print(data)
data['text'] = data['text'].replace(np.nan, '', regex=True)
data['description'] = data['description'].replace(np.nan, '', regex=True)

# drop columns with ground truth
truth = pd.read_csv(r"age_tweets.csv", encoding="utf8")
truth.rename(columns={'Unnamed: 0': '_id'}, inplace=True)


# function to clean the word of any punctuation or special characters
# we need slang and emojis as they reflect the difference between age groups
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


# function to count emojis
def emoji_count(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)

    counter = 0
    datas = list(text)  # split the text into characters

    for word in datas:
        counter += len(emoji_pattern.findall(word))
    return counter


# function to slang words
def slang_count(text):
    slang_data = []
    with open("slang.txt", 'r', encoding="utf8") as exRtFile:
        exchReader = csv.reader(exRtFile, delimiter=':')

        for row in exchReader:
            slang_data.append(row[0].lower())

    counter = 0
    data = text.lower().split()
    for word in data:

        for slang in slang_data:

            if slang == word:
                counter += 1
    return counter


print("Extracting features...")
# count slang and emojis at text and description
data["slang_count"] = ""
data["emoji_count"] = ""
for i in range(50000, 100000):
    data["slang_count"].iloc[i] = slang_count(data['description'].iloc[i])
    data["slang_count"].iloc[i] += slang_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] = emoji_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] += emoji_count(data['description'].iloc[i])

# convert to lower and remove punctuation or special characters

data['description'] = data['description'].str.lower()
data['description'] = data['description'].apply(cleanPunc)

print(data.head(3))

# load tfidf
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3),
                             vocabulary=pickle.load(open("tfidf_age.pkl", 'rb')))
# transform description to tfidf
vectors = vectorizer.fit_transform(data['description'])
vectors_pd = pd.DataFrame(vectors.toarray())

# scale emojis_count and slang_count to [0,1]
scaler = MinMaxScaler()
data[['emoji_count', 'slang_count']] = scaler.fit_transform(data[['emoji_count', 'slang_count']])

# create dataframe X, y for train
X = pd.concat([vectors_pd, data['emoji_count'], data['slang_count']], axis=1)
print(X)

# load classifier
filename = 'adaboost_final.sav'
clf = pickle.load(open(filename, 'rb'))
print("\nPredicting with Adaboost...")

y = clf.predict(X)
y = y.tolist()

data['age_group'] = y
data['original author'] = author_df
data['original_text'] = text_df

# Save original author and original text along with predicted age group
header = ['original author', 'original_text', 'age_group']
data.to_csv('predicted_ages_50k-100k.csv', columns=header)

# Plot the results
counts = data['age_group'].value_counts()

plt.bar(counts.index[:], counts.values[:], color=(0.0, 0.0, 1, 0.5))
plt.title('Twitter users age groups counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()
