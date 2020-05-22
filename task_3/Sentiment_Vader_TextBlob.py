import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
from twitter.connect_mongo import read_mongo




#read data
file1="../dataset/test_cleaned.csv"
insta=pd.read_csv(file1, encoding="utf8")

tweets = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1})
tweets = tweets.sample(n=100, random_state=42)

insta = insta[['_id', 'caption' ]] #this is all I need

vader = SentimentIntensityAnalyzer()

vader_score = []  # empty list to store'compound' VADER scores
textblob_score = []
for i in range(0, len(insta)):
    k = vader.polarity_scores(insta.iloc[i]['caption'])
    l = TextBlob(insta.iloc[i]['caption']).sentiment

    vader_score.append(k)
    textblob_score.append(l)

vader_score=pd.DataFrame.from_dict(vader_score)
textblob_score=pd.DataFrame.from_dict(textblob_score)

insta['VADER compound score'] = vader_score['compound']
insta['TextBlob polarity score']=textblob_score['polarity']

# http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf

pred_V = []
pred_B = []

for i in range(0, len(insta)):
    if ((insta.iloc[i]['VADER compound score'] >= 0.4)):
        pred_V.append('positive')

    elif ((insta.iloc[i]['VADER compound score'] > -0.2) & (insta.iloc[i]['VADER compound score'] < 0.4)):
        pred_V.append('neutral')

    elif ((insta.iloc[i]['VADER compound score'] <= -0.2)):
        pred_V.append('negative')

    if ((insta.iloc[i]['TextBlob polarity score'] >= 0.1)):
        pred_B.append('positive')

    elif ((insta.iloc[i]['TextBlob polarity score'] > -0.1) & (insta.iloc[i]['TextBlob polarity score'] < 0.1)):
        pred_B.append('neutral')

    elif ((insta.iloc[i]['TextBlob polarity score'] <= -0.1)):
        pred_B.append('negative')

insta['VADER predicted sentiment'] = pred_V
insta['TextBlob predicted sentiment'] = pred_B

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
print(insta.head())

#γρρηγορο πλοταριμα
insta.groupby('VADER predicted sentiment').size().plot(kind='bar')
plt.show()
insta.groupby('TextBlob predicted sentiment').size().plot(kind='bar')
plt.show()
