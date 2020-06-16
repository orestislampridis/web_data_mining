import csv
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)

# Read instagram Data
file1 = "../dataset/test_cleaned.csv"
insta = pd.read_csv(file1, encoding="utf8")
insta = insta[['_id','owner_username','caption']]
print("\nInsta posts",insta.head(3))

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
# count slang and emojis at text and description for instagram
insta["slang_count"] = ""
insta["emoji_count"] = ""
# print(data.iloc[0])
for i in range(0,len(insta)):
    insta["slang_count"].iloc[i] = slang_count(insta['caption'].iloc[i])
    insta["emoji_count"].iloc[i] = emoji_count(insta['caption'].iloc[i])

# convert to lower and remove punctuation or special characters
insta['caption'] = insta['caption'].str.lower()
insta['caption'] = insta['caption'].apply(cleanPunc)

print(insta.head(3))
index = insta.index.values #get index(ids) of insta posts


# load tfidf
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3),
                             vocabulary=pickle.load(open("tfidf_age.pkl", 'rb')))

# transform tweets description to tfidf
vectors = vectorizer.fit_transform(insta['caption'])
vectors_pd = pd.DataFrame(vectors.toarray()).set_index(index) # match with index(ids)

# scale tweets emojis_count and slang_count to [0,1]
scaler = MinMaxScaler()
insta[['emoji_count', 'slang_count']] = scaler.fit_transform(insta[['emoji_count', 'slang_count']])

# create dataframe X, y for train tweets
X = pd.concat([vectors_pd, insta['emoji_count'], insta['slang_count']], axis=1)
#print(X.head(3))

# load classifier
filename = 'adaboost_final.sav'
clf = pickle.load(open(filename, 'rb'))
print("\nPredicting with Adaboost...")

y = clf.predict(X)
#print(y)
y = y.tolist()

insta['age_group'] = y


# Save original author and original text along with predicted age group
insta = insta.drop(['emoji_count', 'slang_count'], axis=1)
insta.to_csv('predicted_ages_Instagram.csv')

# Plot the results
counts = insta['age_group'].value_counts()

plt.bar(counts.index[:], counts.values[:], color=(0.9, 0.0, 0.3, 0.8))
plt.title('Instagram users age groups counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()