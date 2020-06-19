# predict likes of a tweet/insta post
import re

import gensim
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as py
from pandas import json_normalize
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

import task_2.preprocessing
import task_6.word2vec_model
from connect_mongo import read_mongo

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
    print(text)
    datas = list(text)  # split the text into characters

    for word in datas:
        counter += len(emoji_pattern.findall(word))
    return counter


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True, only_verbs_nouns=False)

# Read Twitter data
data = read_mongo(db='twitter_db', collection='twitter_collection', query={'retweeted_status': 1}).sample(2000)

# data = data.sample(n=1000, random_state=42)
data = data.dropna()
data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes
print(data)

# get the nested fields screen_name, description from field user
nested_data = json_normalize(data['retweeted_status'])

data['text'] = nested_data['text']
data['favorite_count'] = nested_data['favorite_count']

nested_data['user.description'] = nested_data['user.description'].replace([None], [
    ''])  # replace none values with empty strings

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
data['clean_text'] = data['clean_text'].progress_map(
    lambda clean_text: [word for word in clean_text if word not in more_stopwords])
data['clean_descr'] = data['clean_descr'].progress_map(
    lambda clean_text: [word for word in clean_text if word not in more_stopwords])

# Drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
data.drop(data[data['clean_text'].progress_map(lambda d: len(d)) < 1].index,
          inplace=True)  # drop the rows that contain empty captions
# data[data['clean_text'].str.len() < 1]  # alternative way
data.drop(data[data['clean_descr'].progress_map(lambda d: len(d)) < 1].index,
          inplace=True)  # drop the rows that contain empty descriptions
data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices

# data['clean_txt_descr'] = data['clean_text'] + data['clean_descr']

print(data)  # use to clean non-english posts

# Filter out extremely famous people that destroy the distribution
data = data[data['favorite_count'] < 5000]

# Split the favorite counts into 4 quantiles and get the cut-off points
np.quantile(data.favorite_count, [0, 0.25, 0.5, 0.75, 1])

data['favorite_count_bins'] = pd.qcut(data.favorite_count, 4)
print(data.favorite_count_bins.value_counts().sort_index())

# Split the 'like' field into the four bins defined above, low/normal/high/famous
bin_labels = ['low: (0, 3]', 'normal: (3, 26]', 'high: (26, 233]', 'famous: (233, 4997]']  # class_names
# bin_labels = [0, 1, 2, 3]
data['likes'] = pd.qcut(data['favorite_count'], q=len(bin_labels), labels=bin_labels)

print(data['likes'].value_counts())

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
one_hot = CountVectorizer(lowercase=False, preprocessor=dummy, tokenizer=dummy, analyzer='word')

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

    # ==================================================================================================================
    # Figure comparing best best model performance (Random Forest Classifier is the best)
    # ==================================================================================================================

    if classifier_name == 'Random Forest Classifier' and encoding_name == 'One-hot-encoding':
        report = classification_report(y_true=test, y_pred=prediction, output_dict=True)
        accur = 100 * np.round(report['accuracy'], 4)
        precision = 100 * np.round(report['macro avg']['precision'], 4)
        recall = 100 * np.round(report['macro avg']['recall'], 4)
        fscore = 100 * np.round(report['macro avg']['f1-score'], 4)

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        metrics_scores = [accur, precision, recall, fscore]
        colors = ['cyan', 'crimson', 'coral', 'cadetblue']

        fig = go.Figure(
            data=[go.Bar(x=metrics_names, y=metrics_scores, text='One-hot + RFC performance', marker_color=colors)],
            layout=go.Layout(
                title='Twitter - NLP Like Predicition - Best Model Performance (Random Forest Classifier with One-hot-encoding)',
                yaxis=dict(range=[0, 100],  # sets the range of yaxis
                           constrain="domain")  # meanwhile compresses the yaxis by decreasing its "domain"
            )
            )

        fig.update_yaxes(ticksuffix="%")

        py.plot(fig, filename='twitter_nlp_perform.html')


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

    print(evaluation_scores(y_test, y_predicted, classifier_name=name,
                            encoding_name='TF-IDF'))  # , class_names=class_names

# ======================================================================================================================
# Count Vectorizer
# ======================================================================================================================

for name, classifier in predictors:
    column_trans = ColumnTransformer(
        [('one_hot_text', one_hot, 'clean_text'),
         ('one_hot_descr', one_hot, 'clean_descr')],
        remainder='passthrough')

    X_tfidf_train = column_trans.fit_transform(X_train)
    X_tfidf_test = column_trans.transform(X_test)

    classifier.fit(X_tfidf_train, y_train)
    y_predicted = classifier.predict(X_tfidf_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name,
                            encoding_name='One-hot-encoding'))  # , class_names=class_names

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

    print(evaluation_scores(y_test, y_predicted, classifier_name=name,
                            encoding_name='Word2vec'))  # , class_names=class_names

# ======================================================================================================================
# Plot pie of like distribution
# ======================================================================================================================

# Get our original df to apply the trained classifier and get predictions
predict_df = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'user': 1}).sample(1000)
predict_df = predict_df.dropna()

# get the nested fields screen_name, description from field user
nested_data = json_normalize(predict_df['user'])
print(nested_data.columns.to_list())

predict_data = pd.DataFrame()

predict_data['text'] = predict_df['text']
predict_data['user.description'] = nested_data['description'].replace([None],
                                                                      [''])  # replace none values with empty strings

predict_data["emoji_count"] = ""
for i in range(0, len(predict_data)):
    predict_data["emoji_count"].iloc[i] = emoji_count(predict_data['text'].iloc[i])
    predict_data["emoji_count"].iloc[i] += emoji_count(predict_data['user.description'].iloc[i])

# clean text using preprocessing.py (clean_Text function)
predict_data['clean_descr'] = predict_data['user.description'].progress_map(reading.clean_text)
predict_data['clean_text'] = predict_data.text.progress_map(reading.clean_text)

predict_data['clean_text'] = predict_data['clean_text'].progress_map(
    lambda clean_text: [word for word in clean_text if word not in more_stopwords])
predict_data['clean_descr'] = predict_data['clean_descr'].progress_map(
    lambda clean_text: [word for word in clean_text if word not in more_stopwords])

# Drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
predict_data.drop(predict_data[predict_data['clean_text'].progress_map(lambda d: len(d)) < 1].index,
                  inplace=True)  # drop the rows that contain empty captions
# data[data['clean_text'].str.len() < 1]  # alternative way
predict_data.drop(predict_data[predict_data['clean_descr'].progress_map(lambda d: len(d)) < 1].index,
                  inplace=True)  # drop the rows that contain empty descriptions
predict_data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices

# data['clean_txt_descr'] = data['clean_text'] + data['clean_descr']

print(predict_data)  # use to clean non-english posts

X = predict_data[['clean_text', 'clean_descr', 'emoji_count']]

# Fit best performing model on the whole dataset and predict on it
column_trans = ColumnTransformer(
    [('one_hot_text', one_hot, 'clean_text'),
     ('one_hot_descr', one_hot, 'clean_descr')],
    remainder='passthrough')

X = column_trans.fit_transform(X)

rfc.fit(X, y)

y_predicted = rfc.predict(X)
y_pred = pd.DataFrame(y_predicted, columns=['y_pred'])
print(y_pred)

# Save predicted likes to csv to integrate with map
y_pred.to_csv('predicted_likes.csv')

fig = px.pie(y_pred, names="y_pred", title='Twitter - NLP Like Predicition - Data Distribution on Like count')
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
py.plot(fig, filename='twitter_nlp_pred_best_model.html')
