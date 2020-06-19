# predict likes of a tweet/insta post
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.offline as py
import plotly.graph_objects as go
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
df = read_mongo(db='twitter_db', collection='twitter_collection', query={'retweeted_status': 1}).sample(5000)

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

# Filter out extremely famous people that destroy the distribution
data = data[data['favorite_count'] < 5000]

# Split the favorite counts into 4 quantiles and get the cut-off points
np.quantile(data.favorite_count, [0, 0.25, 0.5, 0.75, 1])

data['favorite_count_bins'] = pd.qcut(data.favorite_count, 4)
print(data.favorite_count_bins.value_counts().sort_index())

# Split the 'like' field into the four bins defined above, low/normal/high/famous
bin_labels = ['low: (0, 3]', 'normal: (3, 22]', 'high: (22, 192]', 'famous: (192, 4919]']  # class_names
# bin_labels = [0, 1, 2, 3]
data['likes'] = pd.qcut(data['favorite_count'], q=len(bin_labels), labels=bin_labels)

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

# predictors = [['LogisticRegression', lr], ['DecisionTreeClassifier', dt], ['SVM', svm], ['Random Forest Classifier', rfc],
#              ['XGB Classifier', xgb_imb_aware]]

predictors = [['Random Forest Classifier', rfc]]


def evaluation_scores(test, prediction, classifier_name=''):
    print('\n', '-' * 60)
    print(classifier_name)
    print('Accuracy:', np.round(accuracy_score(test, prediction), 4))
    print('-' * 60)
    print('classification report:\n\n', classification_report(y_true=test, y_pred=prediction))

    # ==================================================================================================================
    # Figure comparing best best model performance (Random Forest Classifier is the best)
    # ==================================================================================================================

    if classifier_name == 'Random Forest Classifier':
        report = classification_report(y_true=test, y_pred=prediction, output_dict=True)
        accur = 100 * np.round(report['accuracy'], 4)
        precision = 100 * np.round(report['macro avg']['precision'], 4)
        recall = 100 * np.round(report['macro avg']['recall'], 4)
        fscore = 100 * np.round(report['macro avg']['f1-score'], 4)

        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
        metrics_scores = [accur, precision, recall, fscore]
        colors = ['cyan', 'crimson', 'coral', 'cadetblue']

        fig = go.Figure(data=[go.Bar(x=metrics_names, y=metrics_scores, text='RFC performance', marker_color=colors)],
                        layout=go.Layout(
                            yaxis=dict(range=[0, 100],  # sets the range of yaxis
                                       constrain="domain")  # meanwhile compresses the yaxis by decreasing its "domain"
                            )
                        )

        fig.update_layout(title_text='Twitter - Base Model Like Predicition - Best Model Performance (Random Forest Classifier)', title_x=0.5)
        fig.update_yaxes(ticksuffix="%")

        py.plot(fig, filename='twitter_base_model_perform.html')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

for name, classifier in predictors:
    classifier.fit(X_train, y_train)
    y_predicted = classifier.predict(X_test)

    print(evaluation_scores(y_test, y_predicted, classifier_name=name))

# ======================================================================================================================
# Plot pie of like distribution
# ======================================================================================================================

# Get our original df to apply the trained classifier and get predictions
predict_df = read_mongo(db='twitter_db', collection='twitter_collection', query={'user': 1})
predict_df = predict_df.dropna()

# Get the nested fields screen_name, description and other from field user
nested_data = json_normalize(predict_df['user'])
print(nested_data.columns.to_list())

predict_data = pd.DataFrame()

# Feature columns
predict_data['day_of_the_week'] = pd.to_datetime(nested_data["created_at"], errors='ignore', utc=True)
predict_data['day_of_the_week'] = predict_data.day_of_the_week.dt.dayofweek
predict_data['verified'] = nested_data['verified']
predict_data['user_followers_count'] = nested_data['followers_count']
predict_data['user_friends_count'] = nested_data['friends_count']
predict_data['user_favourites_count'] = nested_data['favourites_count']  # Not sure if this is useful
predict_data['user_statuses_count'] = nested_data['statuses_count']

X = predict_data[['day_of_the_week', 'verified', 'user_followers_count', 'user_friends_count', 'user_favourites_count',
                  'user_statuses_count']]

# Fit best performing model on the whole dataset and predict on it
y_predicted = rfc.predict(X)
y_pred = pd.DataFrame(y_predicted, columns=['Predicted Likes'])
print(y_pred)

fig = px.pie(y_pred, names="Predicted Likes")
fig.update_layout(title_text='Twitter - Base Model Like Prediction - Data Distribution on Like count', title_x=0.5)
fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
py.plot(fig, filename='twitter_base_model_pred_best_model.html')


# ======================================================================================================================
# Interpret data features - Feature Importance
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


# ======================================================================================================================
# Interpret data features - Feature Importance with Interactive Chart
# ======================================================================================================================

feature_importances = pd.DataFrame(rfc.feature_importances_,
                                   index=X.columns,
                                   columns=['importance'])
feature_importances['std'] = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
feature_importances.sort_values('importance', ascending=False, inplace=True)
print(feature_importances)


fig = go.Figure(data=[
    go.Bar(name='Feature Importance', x=feature_importances.index, y=feature_importances['importance'],
           error_y=dict(type='data',  # value of error bar given in data coordinates
                        array=feature_importances['std'], visible=True),
           marker={'color': indices, 'colorscale': 'Viridis'})
])

# position the ticks at intervals of dtick=0.25, starting at tick0=0.25
fig.update_layout(xaxis=dict(tick0=0, dtick=0.25))
fig.update_layout(title_text='Twitter - Base Model Like Predicition - Feature Imprortance - Best Model Performance (Random Forest Classifier)', title_x=0.5)

py.plot(fig, filename='twitter_base_model_feature_imp.html')
