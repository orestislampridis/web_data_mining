import pandas as pd
import numpy as np
import re
import pandas as pd
from pandas import json_normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import task_2.preprocessing
from statistics import mean
from tqdm import tqdm
tqdm.pandas()
from connect_mongo import read_mongo

pd.set_option('display.max_columns', None)

#read data instagram
file1 = "../dataset/test_cleaned.csv"
insta = pd.read_csv(file1, encoding="utf8")

print(insta.head(3))

#read data twitter
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'text': 1})
tweets = data.sample(n=1000, random_state=42)

print(tweets.head(3))
#define i-talk
i_talk=['i','me','my','myself','mine']

#a function for simple preprocess
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

insta['caption'] = insta['caption'].map(lambda com : clean_text(com))
tweets['text'] = tweets['text'].map(lambda com : clean_text(com))

#a function to count the occurences of i-talk
def count_occurrences(word, sentence):
    return sentence.split().count(word)

#count i-talk occurances for instagam
score_insta=len(insta)*[0]
for i in range(0,len(insta)):
    for pron in i_talk:
        score_insta[i]+=count_occurrences(pron, insta['caption'].iloc[i])
mean_insta=mean(score_insta) #find mean of i-talk in instagram posts

#count i-talk occurances for twitter
score_tweets=len(tweets)*[0]
for i in range(0,len(tweets)):
    for pron in i_talk:
        score_tweets[i]+=count_occurrences(pron, tweets['text'].iloc[i])
mean_tweets=mean(score_tweets) #find mean of i-talk in tweets

#check if narcissist on instagram
insta['narcissistic']=""
for i in range(0,len(insta)):
    if score_insta[i]>=2*mean_insta:     #non objective threshold!!!
        insta['narcissistic'].iloc[i]="narcissist"
    else:
        insta['narcissistic'].iloc[i]="no_narcissist"

#check if narcissist on twitter
tweets['narcissistic']=""
for i in range(0,len(tweets)):
    if score_tweets[i]>=2*mean_tweets:     #non objective threshold!!!
        tweets['narcissistic'].iloc[i]="narcissist"
    else:
        tweets['narcissistic'].iloc[i]="no_narcissist"

counts_insta = insta['narcissistic'].value_counts()
counts_tweets = tweets['narcissistic'].value_counts()

plt.bar(counts_insta.index[:2], counts_insta.values[:2], color = (0.8,0.0,0.7,0.8))
plt.title('Instagram users extrovert/ introvert counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()

plt.bar(counts_tweets.index[0:2], counts_tweets.values[:2], color = (0.0,0.0,1,0.5))
plt.title('Twitter users extrovert/ introvert counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()
