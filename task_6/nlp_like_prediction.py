# predict likes of a tweet/insta post
import re
import gensim
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as py
import task_2.preprocessing
import task_6.word2vec_model
from sklearn.svm import SVC
from pandas import json_normalize
from xgboost import XGBClassifier
from connect_mongo import read_mongo
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

pd.set_option('display.max_columns', None)

# ======================================================================================================================

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


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)


# Read Twitter data
data = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'retweeted_status': 1})

# Read Insta data
#data = read_mongo(db='Instagram_Data', collection='post_data', query={'text': 1, 'retweeted_status': 1})


#data = data.sample(n=1000, random_state=42)
data = data.dropna()
data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes
print(data)


# get the nested fields screen_name, description from field user
nested_data = json_normalize(data['retweeted_status'])
print(nested_data['user.listed_count'])
data['favourites_count'] = nested_data['user.listed_count']


nested_data['user.description'] = nested_data['user.description'].replace([None], [''])  # replace none values with empty strings



data["emoji_count"] = ""
for i in range(0, len(data)):
    data["emoji_count"].iloc[i] = emoji_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] += emoji_count(nested_data['user.description'].iloc[i])



# clean text using preprocessing.py (clean_Text function)
data['clean_descr'] = nested_data['user.description'].progress_map(reading.clean_text)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.text.progress_map(reading.clean_text)

'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

data.drop(['text'], axis=1, inplace=True)
data.drop(['retweeted_status'], axis=1, inplace=True)


# further filter stopwords
more_stopwords = ['tag', 'not', 'let', 'wait', 'set', 'put', 'add', 'post', 'give', 'way', 'check', 'think', 'www',
                  'must', 'look', 'call', 'minute', 'com', 'thing', 'much', 'happen', 'hahaha', 'quaranotine',
                  'everyone', 'day', 'time', 'week', 'amp', 'find', 'BTu']
data['clean_text'] = data['clean_text'].progress_map(lambda clean_text: [word for word in clean_text if word not in more_stopwords])
data['clean_descr'] = data['clean_descr'].progress_map(lambda clean_text: [word for word in clean_text if word not in more_stopwords])


# Drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
data.drop(data[data['clean_text'].progress_map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty captions
# data[data['clean_text'].str.len() < 1]  # alternative way
data.drop(data[data['clean_descr'].progress_map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty descriptions
data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices


#data['clean_txt_descr'] = data['clean_text'] + data['clean_descr']

print(data)  # use to clean non-english posts


# split the 'like' field into three bins, low/medium/high
bin_labels = ['low', 'medium', 'high']
data['likes'] = pd.qcut(data['favourites_count'], q=3, labels=bin_labels)

print(data['likes'].value_counts())
print(data)

X = data[['clean_text', 'clean_descr', 'emoji_count']]
y = data['likes']

print(X)


# ======================================================================================================================
# Split TRAIN - TEST
# ======================================================================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(X_train)


# ======================================================================================================================


def dummy(token):
    return token

# ======================================================================================================================

# tf-idf
tfidf = TfidfVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, min_df=3, ngram_range=(1, 2))

# one-hot-encoding
one_not = CountVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, analyzer='word')

# word2vec
model = gensim.models.Word2Vec(X_train, size=100, min_count=0, sg=1)
word2vec_vectorizer = dict(zip(model.wv.index2word, model.wv.syn0))


# ======================================================================================================================

# Logistic Regression
lr = LogisticRegression(solver="liblinear", C=300, max_iter=300)

# Decision Tree
dt = DecisionTreeClassifier()

# SVM
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

# XGBClassifier
# scale_pos_weight: scale the gradient for the positive class, set to inverse of the class distribution (ratio 1:5 -> 5)
xgb_imb_aware = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
                            nthread=4, random_state=27)

predictors = [['LinearRegression', lr], ['DecisionTreeClassifier', dt], ['SVM', svm], ['Random Forest Classifier', rfc],
              ['XGB Classifier', xgb_imb_aware]]


# ======================================================================================================================

def evaluation_scores(test, prediction, classifier_name='', encoding_name=''):
    print('\n', '-' * 60)
    print(classifier_name + " + " + encoding_name)
    print('Accuracy:', np.round(accuracy_score(test, prediction), 4))
    print('-' * 60)
    print('classification report:\n\n', classification_report(y_true=test, y_pred=prediction))


# ======================================================================================================================
# TF-IDF
# ======================================================================================================================

for name, classifier in predictors:
    column_trans = ColumnTransformer(
        [('tfidf_text', tfidf, 'clean_text'),
         ('tfidf_descr', tfidf, 'clean_descr')],
    remainder='passthrough')

    X_tfidf_train = column_trans.fit_transform(X_train)
    X_tfidf_test = column_trans.transform(X_test)

    classifier.fit(X_tfidf_train, y_train)
    y_predicted = classifier.predict(X_tfidf_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name, encoding_name='TF-IDF'))  # , class_names=class_names


# ======================================================================================================================
# Count Vectorizer
# ======================================================================================================================

for name, classifier in predictors:
    column_trans = ColumnTransformer(
        [('one_hot_text', one_not, 'clean_text'),
         ('one_hot_descr', one_not, 'clean_descr')],
        remainder='passthrough')

    X_tfidf_train = column_trans.fit_transform(X_train)
    X_tfidf_test = column_trans.transform(X_test)

    classifier.fit(X_tfidf_train, y_train)
    y_predicted = classifier.predict(X_tfidf_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name, encoding_name='One-hot-encoding'))  # , class_names=class_names


# ======================================================================================================================
# Word2vec
# ======================================================================================================================

for name, classifier in predictors:
    column_trans = ColumnTransformer(
        [('tfidf_text', task_6.word2vec_model.TfidfEmbeddingVectorizer(word2vec_vectorizer), 'clean_text'),
         ('tfidf_descr', task_6.word2vec_model.TfidfEmbeddingVectorizer(word2vec_vectorizer), 'clean_descr')],
        remainder='passthrough')

    X_tfidf_train = column_trans.fit_transform(X_train)
    X_tfidf_test = column_trans.transform(X_test)

    classifier.fit(X_tfidf_train, y_train)
    y_predicted = classifier.predict(X_tfidf_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name, encoding_name='Word2vec'))  # , class_names=class_names


# ======================================================================================================================
# Plot pie of like distribution
# ======================================================================================================================

# Fit best performing model on the whole dataset and predict on it
column_trans = ColumnTransformer(
        [('one_hot_text', one_not, 'clean_text'),
         ('one_hot_descr', one_not, 'clean_descr')],
        remainder='passthrough')

X = column_trans.fit_transform(X)

rfc.fit(X, y)
y_predicted = rfc.predict(X)
y_pred = pd.DataFrame(y_predicted, columns=['y_pred'])
print(y_pred)
fig = px.pie(y_pred, names="y_pred")
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
py.plot(fig, filename='twitter_nlp_pred_best_model.html')
