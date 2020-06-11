import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import json_normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from connect_mongo import read_mongo

# Get our initial df with columns 'geo.coordinates' and 'location'
df = read_mongo(db='twitter_db', collection='twitter_collection', query={'retweeted_status': 1}).sample(10000)

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
data['user_listed_count'] = nested_data['user.listed_count']
data['user_favourites_count'] = nested_data['user.favourites_count']  # Not sure if this is useful
data['user_statuses_count'] = nested_data['user.statuses_count']

# Ground truth columns
data['quote_count'] = nested_data['user.friends_count']
data['reply_count'] = nested_data['user.listed_count']
data['retweet_count'] = nested_data['user.friends_count']
data['favorite_count'] = nested_data['user.listed_count']

data.to_csv("ultimate_ground_truth.csv")

# split the 'like' field into three bins, low/medium/high
class_names = ['low', 'medium', 'high']
bin_labels = [0, 1, 2]
data['likes'] = pd.qcut(data['favorite_count'], q=3, labels=bin_labels)

print(data['likes'].value_counts())

X = data[['day_of_the_week', 'verified', 'user_followers_count', 'user_friends_count', 'user_listed_count',
          'user_favourites_count', 'user_statuses_count']]
y = data['likes']


def dummy(token):
    return token


# Linear Regression
lr = LinearRegression()

# Decision Tree
dt = DecisionTreeClassifier()

# SVM
svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)

# Random Forest Classifier
rfc = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rfc.fit(X_train, y_train)
y_predicted = rfc.predict(X_test.values)

importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the impurity-based feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()


def evaluation_scores(test, prediction, class_names=None):
    print('Accuracy:', np.round(accuracy_score(test, prediction), 4))
    print('-' * 60)
    print('classification report:\n\n', classification_report(y_true=test, y_pred=prediction, target_names=class_names))


print(evaluation_scores(y_test, y_predicted, class_names=class_names))
