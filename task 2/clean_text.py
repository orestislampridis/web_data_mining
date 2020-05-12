import string
import re
import string

from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

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

    # remove rt and via in case of tweet data
    tweet = tweet.lower().split()
    tweet = [w for w in tweet]
    tweet = " ".join(tweet)
    tweet = re.sub(r"rt", "", tweet)
    tweet = re.sub(r"via", "", tweet)

    # replace consecutive non-ASCII characters with a space
    tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)

    # remove emojis from tweet
    tweet = emoji_pattern.sub(r'', tweet)
    word_tokens = word_tokenize(tweet)

    filtered_tweet = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words, emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_tweet.append(w)
    return ' '.join(filtered_tweet)


# Method to clean insta posts
def clean_insta_posts(post):
    # remove entities and links
    post = strip_all_entities(strip_links(post))

    # remove repost in case of instagram data
    post = post.lower().split()
    post = [w for w in post]
    post = " ".join(post)
    post = re.sub(r"repost", "", post)

    # replace consecutive non-ASCII characters with a space
    post = re.sub(r'[^\x00-\x7F]+', ' ', post)

    # remove emojis
    post = emoji_pattern.sub(r'', post)
    word_tokens = word_tokenize(post)
    filtered_post = []
    # looping through conditions
    for w in word_tokens:
        # check tokens against stop words, emoticons and punctuations
        if w not in stop_words and w not in emoticons and w not in string.punctuation:
            filtered_post.append(w)
    return ' '.join(filtered_post)
