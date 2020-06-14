import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer

from connect_mongo import read_mongo

data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1, 'text': 1})
pd.set_option('display.max_columns', None)
print(data.head(3))


# print(data.head())
nested_data = json_normalize(data['user'])

author_df = (data['original author'])
desc = (nested_data['description'].tolist())
data['description'] = desc
text_df = (data['text'])

data['text'] = data['text'].replace(np.nan, '', regex=True)
data['description'] = data['description'].replace(np.nan, '', regex=True)

def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


data['description'] = data['description'].str.lower()
data['description'] = data['description'].apply(cleanPunc)
data['text'] = data['text'].str.lower()
data['text'] = data['text'].apply(cleanPunc)


data['content'] = data['text'] + data['description']

# data = pd.concat([author_df, data['content']], axis=1, sort=False)
print(data.head(3))



# # small preprocessing(use of language is important)
# data["content"] = data["content"].str.lower()
# data["content"] = data["content"].apply(cleanPunc)

# print(data.head(3))

# load tfidf
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3),
                             vocabulary=pickle.load(open("tfidf_gender.pkl", 'rb')))
# transform description to tfidf
X = vectorizer.fit_transform(data['content'])

# load classifier
filename = 'clf_final_gender.sav'
clf = pickle.load(open(filename, 'rb'))

print("\nPredicting with {}...".format(clf.__class__.__name__))
y = clf.predict(X)
gender = {1: 'male', 0: 'female'}
y = [gender[item] for item in y]
# print(y)
data['gender'] = y
data['original author'] = author_df
data['original_text'] = text_df
# keep what we need
data = data[['original author', 'original_text', 'gender']]
print("\n Results:")
print(data.head(3))

# Save original author and original text along with predicted age group
header = ['original author', 'original_text', 'gender']
data.to_csv('predicted_genders.csv', columns=header)  # uncomment to save

# Plot the results
counts = data['gender'].value_counts()

plt.bar(counts.index[:], counts.values[:], color=(0.0, 0.0, 1, 0.5))
plt.title('Twitter users gender counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()
