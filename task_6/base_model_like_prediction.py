# predict likes of a tweet/insta post
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from pandas import json_normalize
from connect_mongo import read_mongo
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

pd.set_option('display.max_columns', None)

# Read Twitter data
user_data = read_mongo(db='twitter_db', collection='twitter_collection', query={'user': 1})
user_data = user_data.sample(n=1000, random_state=42)
user_data.reset_index(drop=True, inplace=True)  # needed when we sample the data in order to join dataframes
print(user_data)


# get the nested fields screen_name, description from field user
nested_data = json_normalize(user_data['user'])
print(nested_data['friends_count'])  # favourites_count    verified
print(nested_data['favourites_count'])

data = pd.DataFrame(columns=['verified', 'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count'])
data['verified'] = nested_data['verified']
data['followers_count'] = nested_data['followers_count']
data['friends_count'] = nested_data['friends_count']
data['listed_count'] = nested_data['listed_count']
data['favourites_count'] = nested_data['favourites_count']
data['statuses_count'] = nested_data['statuses_count']

# get the average likes of a user and use it as the target label (because 'like' field for the scrapped post is 0)
data['post_favourites'] = round(nested_data['favourites_count'] / (nested_data['listed_count'] + 1))  # divide with total number of user posts
print(data['post_favourites'])
data['post_favourites'] = data.post_favourites.astype(int)

#nested_data['description'] = nested_data['description'].replace([None], [''])  # replace none values with empty strings


'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices


print(data)  # use to clean non-english posts


# split the 'like' field into three bins, low/medium/high
bin_labels = ['low', 'medium', 'high']
data['likes'] = pd.qcut(data['post_favourites'], q=3, labels=bin_labels)

print(data['likes'].value_counts())

X = data[['verified', 'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count']]
y = data['likes']


def dummy(token):
    return token


# SVM
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='log2', max_depth=None)

# XGBClassifier
# scale_pos_weight: scale the gradient for the positive class, set to inverse of the class distribution (ratio 1:5 -> 5)
xgb_imb_aware = XGBClassifier(learning_rate=0.01, n_estimators=1000, max_depth=4, min_child_weight=6, gamma=0,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.005, objective='binary:logistic',
                            nthread=4, random_state=27)

predictors = [['SVM', svm], ['Random Forest Classifier', rfc], ['XGB Classifier', xgb_imb_aware]]


# set the scoring metrics
scoring = {'precision': 'precision_macro', 'recall': 'recall_macro', 'accuracy': 'accuracy', 'f1-score': 'f1_macro'}


for name, classifier in predictors:
    steps = [('model', classifier)]  # minmax scaler for continuous features
    pipeline = Pipeline(steps=steps)

    # evaluate pipeline
    print("\n", name, ":")
    scores = cross_validate(pipeline, X, y, scoring=scoring, cv=10, return_train_score=False)
    for s in scoring:
        print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))
