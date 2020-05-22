import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import task_2.preprocessing
from tqdm import tqdm
tqdm.pandas()
from twitter.connect_mongo import read_mongo


#get categories
emotion_categories = ['anger', 'joy', 'disgust', 'fear', 'sadness', 'surprise']
#read data
file1="../dataset/test_cleaned.csv"
insta=pd.read_csv(file1, encoding="utf8")

tweets = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1})
tweets = tweets.sample(n=100, random_state=42)




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