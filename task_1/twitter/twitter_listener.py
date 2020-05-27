import json
import string
import sys
import re
import tweepy
import pymongo
from datetime import datetime
from nltk import word_tokenize
from textblob import TextBlob
import datetime
from nltk.corpus import stopwords

# Authorization tokens
consumer_key = 'v0xKMKsBMFN5h2WUmWTG1leh8'
consumer_secret = 'rsSy7BfKhXU61ktvbn7VF9SHbCcTNZJ65xcvYWcc8dLhzAEbuY'
access_key = '133859328-QITghlxAxmVaDJim41H7hxmDSzTUk2pusFVPc6sS'
access_secret = 'ZJUF0Enx27RYltuz2cB7ItFxhinBlZx38PinZqEvmae5T'

# Connect to mongodb
client = pymongo.MongoClient('localhost', 27017)
db = client['twitter_db']
collection = db['twitter_collection']

# Happy Emoticons
emoticons_happy = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}', ':^)', ':-D', ':D', '8-D',
                   '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D', '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P',
                   ':-P', ':P', 'X-P', 'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)', '<3'}

# Sad Emoticons
emoticons_sad = {':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<', ':-[', ':-<', '=\\', '=/',
                 '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c', ':c', ':{', '>:\\', ';('}

# Emoji patterns
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

# Combine sad and happy emoticons
emoticons = emoticons_happy.union(emoticons_sad)


def strip_links(text):
    link_regex = re.compile('((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)', re.DOTALL)
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ', ')
    return text


def strip_all_entities(text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


# Method to clean tweets
def clean_tweets(tweet):
    # remove entities and links
    tweet = strip_all_entities(strip_links(tweet))

    # remove rt
    tweet = tweet.lower().split()
    tweet = [w for w in tweet]
    tweet = " ".join(tweet)
    tweet = re.sub(r"rt", "", tweet)

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    word_tokens = word_tokenize(tweet)

    stop_words = set(stopwords.words('english'))
    filtered_tweet = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words, emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


# StreamListener class inherits from tweepy.StreamListener and overrides on_status/on_error methods.
class StreamListener(tweepy.StreamListener):
    def on_data(self, data):

            print(data)
            tweet = json.loads(data)
            print(tweet)
            # Call clean_tweet method for text preprocessing
            filtered_tweet = clean_tweets(tweet['text'])

            # Pass textBlob method for sentiment calculation
            blob = TextBlob(filtered_tweet)
            Sentiment = blob.sentiment

            # Separate polarity and subjectivity in to two variables
            polarity = Sentiment.polarity
            subjectivity = Sentiment.subjectivity

            # Grab the 'created_at' data from the Tweet to use for display and change it to Date object
            created_at = tweet['created_at']
            dt = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
            tweet['created_at'] = dt

            # New entries append
            y1 = {"filtered_tweet": filtered_tweet}
            tweet.update(y1)
            y2 = {"sentiment": Sentiment}
            tweet.update(y2)
            y3 = {"polarity": polarity}
            tweet.update(y3)
            y4 = {"subjectivity": subjectivity}
            tweet.update(y4)

            # Append original author of the tweet
            y5 = {"original author": tweet['user']['screen_name']}
            tweet.update(y5)

            try:
                is_sensitive = tweet['possibly_sensitive']
            except KeyError:
                is_sensitive = None

            y6 = {"is_sensitive": is_sensitive}
            tweet.update(y6)

            # Get hashtags and mentions using comma separated
            hashtags = ", ".join([hashtag_item['text'] for hashtag_item in tweet['entities']['hashtags']])
            y7 = {"hashtags": hashtags}
            tweet.update(y7)
            mentions = ", ".join([mention['screen_name'] for mention in tweet['entities']['user_mentions']])
            y8 = {"mentions": mentions}
            tweet.update(y8)

            # Get location of the tweet if possible
            try:
                location = tweet['user']['location']
            except TypeError:
                location = None
            y9 = {"location": location}
            tweet.update(y9)

            # Get coordinates of the tweet if possible
            try:
                coordinates = [coord for loc in tweet['place']['bounding_box']['coordinates'] for coord in loc]
            except TypeError:
                coordinates = None
            y10 = {"coordinates": coordinates}
            tweet.update(y10)

            print(tweet)
            collection.insert(tweet)
            print('tweet inserted')

    def on_error(self, status_code):
        print("Encountered streaming error (", status_code, ")")
        sys.exit()


if __name__ == "__main__":
    # complete authorization and initialize API endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    # initialize stream
    streamListener = StreamListener()
    stream = tweepy.Stream(auth=api.auth, listener=streamListener,tweet_mode='extended')
    tags = ['#quarantine']
    stream.filter(languages=["en"], track=tags)
