# predict likes of a tweet/insta post
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as py
from pandas import json_normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from connect_mongo import read_mongo

pd.set_option('display.max_columns', None)

# Get our initial df with columns 'geo.coordinates' and 'location'
df = read_mongo(db='twitter_db', collection='twitter_collection', query={'retweeted_status': 1})#.sample(10000)

# Read Insta data
#data = read_mongo(db='Instagram_Data', collection='post_data', query={'text': 1, 'retweeted_status': 1})

df = df.dropna()

# Get the nested fields screen_name, description from field user
nested_data = json_normalize(df['retweeted_status'])
print(nested_data.columns.to_list())

data = pd.DataFrame()

# Feature columns
# data['text'] = nested_data['text']
data['day_of_the_week'] = pd.to_datetime(nested_data["created_at"], errors='ignore', utc=True)
data['day_of_the_week'] = data.day_of_the_week.dt.dayofweek
# data['user_mentions'] = nested_data['entities.user_mentions']
# data['hashtags'] = nested_data['entities.hashtags']
data['verified'] = nested_data['user.verified']
data['user_followers_count'] = nested_data['user.followers_count']
data['user_friends_count'] = nested_data['user.friends_count']
data['user_favourites_count'] = nested_data['user.favourites_count']  # Not sure if this is useful
data['user_statuses_count'] = nested_data['user.statuses_count']

# Ground truth columns
data['quote_count'] = nested_data['quote_count']
data['reply_count'] = nested_data['reply_count']
data['retweet_count'] = nested_data['retweet_count']
data['favorite_count'] = nested_data['favorite_count']

# data.to_csv("ultimate_ground_truth.csv")

# split the 'like' field into three bins, low/medium/high
bin_labels = ['low', 'medium', 'high']  # class_names
# bin_labels = [0, 1, 2]
data['likes'] = pd.qcut(data['favorite_count'], q=3, labels=bin_labels)

print(data['likes'].value_counts())

X = data[['day_of_the_week', 'verified', 'user_followers_count', 'user_friends_count', 'user_favourites_count',
          'user_statuses_count']]
y = data['likes']

print(X)
print(y)


def dummy(token):
    return token


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


def evaluation_scores(test, prediction, classifier_name='', class_names=None):
    print('\n', '-' * 60)
    print(classifier_name)
    print('Accuracy:', np.round(accuracy_score(test, prediction), 4))
    print('-' * 60)
    print('classification report:\n\n', classification_report(y_true=test, y_pred=prediction))
    # print('classification report:\n\n', classification_report(y_true=test, y_pred=prediction, target_names=class_names))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for name, classifier in predictors:
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name))  # , class_names=class_names


# ======================================================================================================================
# Plot pie of like distribution
# ======================================================================================================================

# Fit best performing model on the whole dataset and predict on it
rfc.fit(X, y)
y_predicted = rfc.predict(X)
y_pred = pd.DataFrame(y_predicted, columns=['y_pred'])
print(y_pred)
fig = px.pie(y_pred, names="y_pred")
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
py.plot(fig, filename='twitter_base_line_pred_best_model.html')


# ======================================================================================================================
# Interpret data features
# ======================================================================================================================

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
