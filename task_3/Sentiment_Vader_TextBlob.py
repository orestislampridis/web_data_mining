import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

from connect_mongo import read_mongo




#read data
file1="../dataset/test_cleaned.csv"
insta=pd.read_csv(file1, encoding="utf8")

tweets = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1})
#tweets = tweets.sample(n=1000, random_state=42)



insta = insta[['_id', 'caption' ]] #this is all I need
#tweets= tweets[['text']]

vader = SentimentIntensityAnalyzer()

#check Vader and TextBlob sentiment score for INSTAGRAM POSTS
vader_score = []  # empty list to store'compound' VADER scores
textblob_score = [] # empty list to store 'polarity' TextBlob scores
for i in range(0, len(insta)):
    k = vader.polarity_scores(insta.iloc[i]['caption'])
    l = TextBlob(insta.iloc[i]['caption']).sentiment

    vader_score.append(k)
    textblob_score.append(l)

vader_score_insta=pd.DataFrame.from_dict(vader_score)
textblob_score_insta=pd.DataFrame.from_dict(textblob_score)
#print(textblob_score_insta['polarity'])
insta['VADER compound score'] = vader_score_insta['compound']
insta['TextBlob polarity score']=textblob_score_insta['polarity']
#print(insta['TextBlob polarity score'])



#check Vader and TextBlob sentiment score for TWEETS
vader_score = []  # empty list to store'compound' VADER scores
textblob_score = [] # empty list to store 'polarity' TextBlob scores
for i in range(0, len(tweets)):
    k = vader.polarity_scores(tweets.iloc[i]['text'])
    l = TextBlob(tweets.iloc[i]['text']).sentiment

    vader_score.append(k)
    textblob_score.append(l)

vader_score_tweets=pd.DataFrame.from_dict(vader_score)
textblob_score_tweets=pd.DataFrame.from_dict(textblob_score)

tweets['VADER compound score'] = vader_score_tweets['compound'].to_list()
tweets['TextBlob polarity score']=textblob_score_tweets['polarity'].to_list()



#Predict sentiment in INSTAGRAM POSTS with Vader and TextBlob
pred_V_insta = []
pred_B_insta = []

for i in range(0, len(insta)):
    if ((insta.iloc[i]['VADER compound score'] >= 0.4)):
        pred_V_insta.append('positive')

    elif ((insta.iloc[i]['VADER compound score'] > -0.2) & (insta.iloc[i]['VADER compound score'] < 0.4)):
        pred_V_insta.append('neutral')

    elif ((insta.iloc[i]['VADER compound score'] <= -0.2)):
        pred_V_insta.append('negative')

    if ((insta.iloc[i]['TextBlob polarity score'] >= 0.1)):
        pred_B_insta.append('positive')

    elif ((insta.iloc[i]['TextBlob polarity score'] > -0.1) & (insta.iloc[i]['TextBlob polarity score'] < 0.1)):
        pred_B_insta.append('neutral')

    elif ((insta.iloc[i]['TextBlob polarity score'] <= -0.1)):
        pred_B_insta.append('negative')

insta['VADER predicted sentiment'] = pred_V_insta
insta['TextBlob predicted sentiment'] = pred_B_insta




#Predict sentiment in TWEETS with Vader and TextBlob
pred_V_tweets = []
pred_B_tweets = []

for i in range(0, len(tweets)):
    if ((tweets.iloc[i]['VADER compound score'] >= 0.4)):
        pred_V_tweets.append('positive')

    elif ((tweets.iloc[i]['VADER compound score'] > -0.2) & (tweets.iloc[i]['VADER compound score'] < 0.4)):
        pred_V_tweets.append('neutral')

    elif ((tweets.iloc[i]['VADER compound score'] <= -0.2)):
        pred_V_tweets.append('negative')

    if ((tweets.iloc[i]['TextBlob polarity score'] >= 0.1)):
        pred_B_tweets.append('positive')

    elif ((tweets.iloc[i]['TextBlob polarity score'] > -0.1) & (tweets.iloc[i]['TextBlob polarity score'] < 0.1)):
        pred_B_tweets.append('neutral')

    elif ((tweets.iloc[i]['TextBlob polarity score'] <= -0.1)):
        pred_B_tweets.append('negative')


tweets['VADER predicted sentiment'] = pred_V_tweets
tweets['TextBlob predicted sentiment'] = pred_B_tweets




pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)
print("Instagram sentiment predictions")
print(insta.head())
print("\nTwitter sentiment predictions")
print(tweets.head())


#Pie plots for INSTAGRAM
fig = plt.figure(figsize=(10,10))
ax_insta_V = fig.add_axes([0,0,1,1])
ax_insta_V.axis('equal')
colors = ['#ff9999','#66b3ff','#99ff99']
ax_insta_V.pie(insta.groupby('VADER predicted sentiment').size(),colors=colors, autopct='%1.1f%%',labels=insta.groupby('VADER predicted sentiment').size().index, shadow=True, startangle=90,textprops={'fontsize': 14})
plt.title('Instagram/n VADER predicted sentiment', bbox={'facecolor':'0.8', 'pad':5})
plt.show()

fig = plt.figure(figsize=(10,10))

ax_insta_B = fig.add_axes([0,0,1,1])
ax_insta_B.axis('equal')
colors = ['#ff9999','#66b3ff','#99ff99']
ax_insta_B.pie(insta.groupby('TextBlob predicted sentiment').size(),colors=colors, autopct='%1.1f%%',labels=insta.groupby('VADER predicted sentiment').size().index, shadow=True, startangle=90,textprops={'fontsize': 14})
plt.title('Instagram/n TextBlob predicted sentiment', bbox={'facecolor':'0.8', 'pad':5})
plt.show()


#Pie plots for TWITTER
fig = plt.figure(figsize=(10,10))
ax_insta_V = fig.add_axes([0,0,1,1])
ax_insta_V.axis('equal')
colors = ['#ff9999','#66b3ff','#99ff99']
ax_insta_V.pie(tweets.groupby('VADER predicted sentiment').size(),colors=colors, autopct='%1.1f%%',labels=tweets.groupby('VADER predicted sentiment').size().index, shadow=True, startangle=90,textprops={'fontsize': 14})
plt.title('Instagram/n VADER predicted sentiment', bbox={'facecolor':'0.8', 'pad':5})
plt.show()

fig = plt.figure(figsize=(10,10))

ax_insta_B = fig.add_axes([0,0,1,1])
ax_insta_B.axis('equal')
colors = ['#ff9999','#66b3ff','#99ff99']
ax_insta_B.pie(tweets.groupby('TextBlob predicted sentiment').size(),colors=colors, autopct='%1.1f%%',labels=tweets.groupby('VADER predicted sentiment').size().index, shadow=True, startangle=90,textprops={'fontsize': 14})
plt.title('Instagram/n TextBlob predicted sentiment', bbox={'facecolor':'0.8', 'pad':5})
plt.show()

#save sentiment Tweets for later task
tweets.to_csv('sentiment_tweets.csv')