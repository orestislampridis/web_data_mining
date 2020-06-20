import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.graph_objects as go
import task_2.preprocessing
from tqdm import tqdm
tqdm.pandas()
from connect_mongo import read_mongo


#get categories
emotion_categories = ['anger', 'joy', 'disgust', 'fear', 'sadness', 'surprise']
#read data
insta = pd.read_csv("../dataset/insta_data_cleaned.csv", sep='~', index_col=False)

tweets = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'created_at': 1})


reading = task_2.preprocessing.preprocessing(convert_lower=False, use_spell_corrector=True)
insta['caption'] = insta.caption.progress_map(reading.clean_text)
tweets['text']=tweets.text.progress_map(reading.clean_text)

#print(SemEval['Tweet'].iloc[:2])
def list2string(list):
        return ','.join(map(str, list))

insta['caption'] = [list2string(list) for list in insta['caption']]
tweets['text'] = [list2string(list) for list in tweets['text']]



#predict affective Instagram
pred_insta = []
for category in emotion_categories:
    filename = 'Task 3 Classifiers\OneVsRest_SVC_{}'.format(category)
    classifier = loaded_model = pickle.load(open(filename, 'rb'))
    pred = classifier.predict(insta['caption'])
    pred_insta.append(pred)
#create dataframe for predictions
pred_insta = np.array(pred_insta)
pred_insta = np.transpose(pred_insta)
pred_insta = pd.DataFrame(data=pred_insta, columns=emotion_categories)
result_insta = pd.concat([insta, pred_insta], axis=1, sort=False)

#predict affective Twitter
pred_twitter = []
for category in emotion_categories:
    filename = 'Task 3 Classifiers\OneVsRest_SVC_{}'.format(category)
    classifier = loaded_model = pickle.load(open(filename, 'rb'))
    pred = classifier.predict(tweets['text'])
    pred_twitter.append(pred)
#create dataframe for predictions
pred_twitter = np.array(pred_twitter)
pred_twitter = np.transpose(pred_twitter)
pred_twitter = pd.DataFrame(data=pred_twitter, columns=emotion_categories)
result_twitter = pd.concat([tweets,pred_twitter], axis=1, sort=False)



print("result_insta", result_insta)
print(result_insta.columns.to_list())
print("result_twitter", result_twitter)
print(result_twitter.columns.to_list())
print(emotion_categories)


# ======================================================================================================================
# Create time series
# ======================================================================================================================

# USED FOR PLOTTING
result_insta.rename(columns={'date': 'created_at'}, inplace=True)  # rename column date to created_at
result_insta = result_insta.sort_values(by=['created_at'])

result_insta['datetime'] = pd.to_datetime(result_insta['created_at'])
result_insta = result_insta.set_index('datetime')
result_insta.drop(['created_at'], axis=1, inplace=True)
print(result_insta.columns)


# ======================================================================================================================
# Plot in interactive graph the number of extracted post per time period
# ======================================================================================================================

print("data.index.unique() ", result_insta.index.map(lambda t: t.date()).unique())


fig = go.Figure()
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['anger'],
                    mode='lines+markers',
                    name='anger'))
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['joy'],
                    mode='lines+markers',
                    name='joy'))
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['disgust'],
                    mode='lines+markers',
                    name='disgust'))
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['fear'],
                    mode='lines+markers',
                    name='fear'))
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['sadness'],
                    mode='lines+markers',
                    name='sadness'))
fig.add_trace(go.Scatter(x=result_insta.index.map(lambda t: t.date()).unique(), y=result_insta['surprise'],
                    mode='lines+markers',
                    name='surprise'))

py.plot(fig, filename='insta_emotion_per_day.html')


# ======================================================================================================================
# Create time series
# ======================================================================================================================

# USED FOR PLOTTING
result_twitter = result_twitter.sort_values(by=['created_at'])

result_twitter['datetime'] = pd.to_datetime(result_twitter['created_at'])
result_twitter = result_twitter.set_index('datetime')
result_twitter.drop(['created_at'], axis=1, inplace=True)
print(result_twitter.columns)


# ======================================================================================================================
# Plot in interactive graph the number of extracted post per time period
# ======================================================================================================================

print("data.index.unique() ", result_twitter.index.map(lambda t: t.date()).unique())


fig = go.Figure()
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['anger'],
                    mode='lines+markers',
                    name='anger'))
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['joy'],
                    mode='lines+markers',
                    name='joy'))
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['disgust'],
                    mode='lines+markers',
                    name='disgust'))
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['fear'],
                    mode='lines+markers',
                    name='fear'))
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['sadness'],
                    mode='lines+markers',
                    name='sadness'))
fig.add_trace(go.Scatter(x=result_twitter.index.map(lambda t: t.date()).unique(), y=result_twitter['surprise'],
                    mode='lines+markers',
                    name='surprise'))

py.plot(fig, filename='twitter_emotion_per_day.html')


# ======================================================================================================================


#plot results Instagram
plt.figure(figsize=(15,8))
sums=result_insta.iloc[:,-6:].sum().values
ax= sns.barplot(emotion_categories,sums )
plt.title("Instagram posts in each category", fontsize=22)
plt.ylabel('Number of posts', fontsize=15)
plt.xlabel('Post type ', fontsize=15)
plt.show()

rowsums = result_insta.iloc[:,-6:].sum(axis=1).value_counts()
plt.figure(figsize=(15,8))
ax = sns.barplot(rowsums.index, rowsums.values)
plt.title("Categories per Instagram post", fontsize=22)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of categories', fontsize=15)
plt.show()

#plot results Twitter
plt.figure(figsize=(15,8))
sums=result_twitter.iloc[:,-6:].sum().values
ax= sns.barplot(emotion_categories,sums )
plt.title("Tweets in each category", fontsize=22)
plt.ylabel('Number of tweets', fontsize=15)
plt.xlabel('Tweet type ', fontsize=15)
plt.show()

rowsums = result_twitter.iloc[:,-6:].sum(axis=1).value_counts()
plt.figure(figsize=(15,8))
ax = sns.barplot(rowsums.index, rowsums.values)
plt.title("Categories per Tweet", fontsize=22)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of categories', fontsize=15)
plt.show()