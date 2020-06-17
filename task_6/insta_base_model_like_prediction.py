# predict likes of a tweet/insta post
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import plotly.express as px
import plotly.offline as py
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)


all_data = pd.read_csv("../dataset/insta_data_cleaned.csv", sep='~', index_col=False)

all_data = all_data.dropna()

data = pd.DataFrame()

# Feature columns
# data['text'] = nested_data['text']
data['day_of_the_week'] = pd.to_datetime(all_data["date"], errors='ignore', utc=True)
data['day_of_the_week'] = data.day_of_the_week.dt.dayofweek
data['verified'] = all_data['owner_verified']
data['followers'] = all_data['owner_followers']
data['owner_private'] = all_data['owner_private']
data['owner_viewable_story'] = all_data['owner_viewable_story']
data['total_posts'] = all_data['total_posts']
data['videos_count_all'] = all_data['videos_count_all']
data['photos_count_all'] = all_data['photos_count_all']
data['all_photos_avg_likes'] = all_data['all_photos_avg_likes']
data['all_photos_stdev_likes'] = all_data['all_photos_stdev_likes']
data['all_photos_avg_comments'] = all_data['all_photos_avg_comments']
data['all_photos_stdev_comments'] = all_data['all_photos_stdev_comments']
data['all_videos_avg_likes'] = all_data['all_videos_avg_likes']
data['all_videos_stdev_likes'] = all_data['all_videos_stdev_likes']
data['all_videos_avg_comments'] = all_data['all_videos_avg_comments']
data['all_videos_stdev_comments'] = all_data['all_videos_stdev_comments']
data['all_videos_avg_views'] = all_data['all_videos_avg_views']
data['all_videos_stdev_views'] = all_data['all_videos_stdev_views']
data['followees'] = all_data['followees']
data['caption_hashtags'] = all_data['caption_hashtags'].apply(lambda x: len(x))
data['caption_mentions'] = all_data['caption_mentions'].apply(lambda x: len(x))
data['tagged_users'] = all_data['tagged_users'].apply(lambda x: len(x))


# Ground truth columns
data['likes'] = all_data['likes']

# data.to_csv("ultimate_ground_truth.csv")

# split the 'like' field into three bins, low/medium/high
bin_labels = ['low', 'medium', 'high']  # class_names
# bin_labels = [0, 1, 2]
data['likes'] = pd.qcut(data['likes'], q=3, labels=bin_labels)

print(data['likes'].value_counts())

y = data['likes']
data.drop('likes', axis=1, inplace=True)
X = data


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
py.plot(fig, filename='insta_base_line_pred_best_model.html')


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
