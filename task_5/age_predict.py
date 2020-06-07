
import matplotlib.pyplot as plt
import pandas as pd
from pandas import json_normalize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from connect_mongo import read_mongo
import pickle
import csv
import re
import numpy as np

# Read Twitter data
# description: bio
# original author: username
# screen_name: full user name
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1,'text': 1})
pd.set_option('display.max_columns', None)

nested_data = json_normalize(data['user'])

author_df = (data['original author'])
desc_df = (nested_data['description'])
text_df = (data['text'])

data = pd.concat([author_df, desc_df,text_df], axis=1, sort=False)
data['description']=data['description'].replace(np.nan, '', regex=True)

#drop columns with ground truth
truth = pd.read_csv(r"age_tweets.csv", encoding="utf8")
truth.rename(columns={'Unnamed: 0':'_id'}, inplace=True)
data=data.drop(truth['_id'].values[:2].tolist())

print(data.head())

# function to clean the word of any punctuation or special characters
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

#count slang and emojis at text and description
data["slang_count"] = ""
data["emoji_count"] = ""
for i in range(0,len(data)):

    data["slang_count"].iloc[i] = slang_count(data['description'].iloc[i])
    data["slang_count"].iloc[i] += slang_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] = emoji_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] += emoji_count(data['description'].iloc[i])

#convert to lower and remove punctuation or special characters

data['description']=data['description'].str.lower()
data['description']=data['description'].apply(cleanPunc)

print(data.head())



#load tfidf
vectorizer = TfidfVectorizer(stop_words='english',max_features=10000,ngram_range=(1,3), vocabulary = pickle.load(open("tfidf_age.pkl", 'rb')))
#transform description to tfidf
vectors=vectorizer.fit_transform(data['description'])
vectors_pd=pd.DataFrame(vectors.toarray())

#scale emojis_count and slang_count to [0,1]
scaler = MinMaxScaler()
data[['emoji_count', 'slang_count']] = scaler.fit_transform(data[['emoji_count', 'slang_count']])

#create dataframe X, y for train
X=pd.concat([vectors_pd,data['emoji_count'],data['slang_count']],axis=1)

#load classifier
filename = 'adaboost_final.sav'
clf = loaded_model = pickle.load(open(filename, 'rb'))

y=clf.predict(X)

pred_age = pd.DataFrame(data=y, columns=age_group)

result_age = pd.concat([data,pred_age], axis=1, sort=False)
print(result_age)

#plots
result_age.age_group.plot(kind='hist')
plt.show()

result_age.age_group.plot(kind='kde', xlim=(0, 100), grid=True)
plt.show()