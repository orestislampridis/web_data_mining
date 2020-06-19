
import matplotlib.pyplot as plt
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm

tqdm.pandas()
from connect_mongo import read_mongo

pd.set_option('display.max_columns', None)

# read data instagram
file1 = "../dataset/insta_data_cleaned.csv"
insta = pd.read_csv(file1, sep='~', encoding="utf8")
insta = insta.drop_duplicates(subset='owner_id', keep="first")
insta = insta.reset_index(drop=True)
insta = insta[['_id', 'owner_id', 'owner_followers', 'followees', 'comments', 'likes', 'total_posts']]

print(insta.head(3))

# read data twitter
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1})
data = data.sample(n=1000, random_state=42)
# get the nested fields from field user
tweets = json_normalize(data['user'])
tweets = tweets[['id', 'followers_count', 'friends_count', 'favourites_count', 'statuses_count']]
print(tweets.head(3))

# find mean
mean_insta = insta[["owner_followers", "followees", "comments", "likes", "total_posts"]].mean()
mean_tweets = tweets[['followers_count', 'friends_count', 'favourites_count', 'statuses_count']].mean()

print(mean_insta)
print(mean_tweets)

score_insta = len(insta) * [0]
print(len(insta))
print(len(score_insta))
for i in range(0, len(insta)):

    if insta['owner_followers'].iloc[i] >= mean_insta['owner_followers']:
        score_insta[i] += 1

    if insta['followees'].iloc[i] >= mean_insta['followees']:
        score_insta[i] += 1

    if insta['comments'].iloc[i] >= mean_insta['comments']:
        score_insta[i] += 1

    if insta['likes'].iloc[i] >= mean_insta['likes']:
        score_insta[i] += 1

    if insta['total_posts'].iloc[i] >= mean_insta['total_posts']:
        score_insta[i] += 1

score_tweets = len(tweets) * [0]
for i in range(0, len(tweets)):

    if tweets['followers_count'].iloc[i] >= mean_tweets['followers_count']:
        score_tweets[i] += 1

    if tweets['friends_count'].iloc[i] >= mean_tweets['friends_count']:
        score_tweets[i] += 1

    if tweets['favourites_count'].iloc[i] >= mean_tweets['favourites_count']:
        score_tweets[i] += 1

    if tweets['statuses_count'].iloc[i] >= mean_tweets['statuses_count']:
        score_tweets[i] += 1

insta['intro/extro'] = " "
for i in range(0, len(insta)):

    if score_insta[i] >= 2:
        insta['intro/extro'][i] = 'extrovert'
    else:
        insta['intro/extro'][i] = 'introvert'

tweets['intro/extro'] = " "
for i in range(0, len(tweets)):

    if score_tweets[i] >= 2:
        tweets['intro/extro'][i] = 'extrovert'
    else:
        tweets['intro/extro'][i] = 'introvert'

counts_insta = insta['intro/extro'].value_counts()
counts_tweets = tweets['intro/extro'].value_counts()

plt.bar(counts_insta.index[:2], counts_insta.values[:2], color = (0.8,0.0,0.7,0.8))
plt.title('Instagram users extrovert/ introvert counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()

plt.bar(counts_tweets.index[:2], counts_tweets.values[:2], color=(0.0, 0.0, 1, 0.5))
plt.title('Twitter users extrovert/ introvert counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()