# predict likes of a tweet/insta post
import pandas as pd
import task_2.preprocessing
from sklearn.svm import SVC
from pandas import json_normalize
from connect_mongo import read_mongo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)


# Read Twitter data
data = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'user': 1})
data = data.sample(n=1000, random_state=42)
data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes
print(data)


# get the nested fields screen_name, description from field user
nested_data = json_normalize(data['user'])
print(nested_data['friends_count'])  # favourites_count    verified
print(nested_data['favourites_count'])
data['favourites_count'] = nested_data['favourites_count']

#nested_data['description'] = nested_data['description'].replace([None], [''])  # replace none values with empty strings


# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.text.progress_map(reading.clean_text)

'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

data.drop(['text'], axis=1, inplace=True)
data.drop(['user'], axis=1, inplace=True)


# further filter stopwords
more_stopwords = ['tag', 'not', 'let', 'wait', 'set', 'put', 'add', 'post', 'give', 'way', 'check', 'think', 'www',
                  'must', 'look', 'call', 'minute', 'com', 'thing', 'much', 'happen', 'hahaha', 'quaranotine',
                  'everyone', 'day', 'time', 'week', 'amp', 'find', 'BTu']
data['clean_text'] = data['clean_text'].progress_map(lambda clean_text: [word for word in clean_text if word not in more_stopwords])


# Drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
data.drop(data[data['clean_text'].progress_map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty captions
# data[data['clean_text'].str.len() < 1]  # alternative way
data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices


print(data)  # use to clean non-english posts


# split the 'like' field into three bins, low/medium/high
bin_labels = ['low', 'medium', 'high']
data['likes'] = pd.qcut(data['favourites_count'], q=3, labels=bin_labels)
print(data['likes'].value_counts())

X = data['clean_text']
y = data['likes']


def dummy(token):
    return token


tfidf = TfidfVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, min_df=3, ngram_range=(1, 2))

# Class imbalance aware
clf2 = LogisticRegression(solver="liblinear", C=400, max_iter=300)

svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

predictors = [['LogisticRegression', clf2], ['SVM', svm]]

scoring = {'precision': 'precision_macro', 'recall': 'recall_macro', 'accuracy': 'accuracy', 'f1-score': 'f1_macro'}


for name, classifier in predictors:
    steps = [('tfidf', tfidf), ('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\n", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
