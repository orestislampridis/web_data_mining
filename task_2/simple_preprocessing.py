# pre-process and clean data
import re
import string

from nltk import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import *  # from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer  # used for lemmatizer


# nltk.download('wordnet')


# ======================================================================================================================
# Remove Contractions (pre-processing)
# ======================================================================================================================

def get_contractions():
    contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                        "could've": "could have", "couldn't": "could not", "didn't": "did not",
                        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                        "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                        "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                        "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                        "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                        "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                        "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                        "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                        "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                        "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                        "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                        "she'll've": "she will have", "she's": "she is", "should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                        "so's": "so as", "this's": "this is", "that'd": "that would",
                        "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                        "there'd've": "there would have", "there's": "there is", "here's": "here is",
                        "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                        "they'll've": "they will have", "they're": "they are", "they've": "they have",
                        "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                        "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                        "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                        "when've": "when have", "where'd": "where did", "where's": "where is",
                        "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                        "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                        "will've": "will have", "won't": "will not", "won't've": "will not have",
                        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                        "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                        "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                        "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                        "you're": "you are", "you've": "you have", "nor": "not", "nt": "not"}

    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


def replace_contractions(text):
    contractions, contractions_re = get_contractions()

    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


whitelist = ["n't", "not", 'nor', "nt"]  # Keep the words "n't" and "not", 'nor' and "nt"
stopwords_verbs = ['say', 'get', 'go', 'know', 'may', 'need', 'like', 'make', 'see', 'want', 'come', 'take', 'use',
                   'would', 'can']
stopwords_other = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also',
                   'across', 'among', 'beside', 'however', 'yet', 'within', 'mr', 'bbc', 'image', 'getty', 'de', 'en',
                   'caption', 'copyright', 'something']
stop_words = set(list(stopwords.words('english')) + ['"', '|'] + stopwords_verbs + stopwords_other)

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
        if separator not in entity_prefixes:
            text = text.replace(separator, ' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)


# convert POS tag to wordnet tag in order to use in lemmatizer
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# function for lemmatazing
def lemmatizing(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    lemma_text = []

    # annotate words with Part-of-Speech tags, format: ((word1, post_tag), (word2, post_tag), ...)
    word_pos_tag = pos_tag(tokenized_text)
    # print("word_pos_tag", word_pos_tag)

    for word_tag in word_pos_tag:  # word_tag[0]: word, word_tag[1]: tag
        # Lemmatizing each word with its POS tag, in each sentence
        if get_wordnet_pos(
                word_tag[1]) != '':  # if the POS tagger annotated the given word, lemmatize the word using its POS tag
            lemma = lemmatizer.lemmatize(word_tag[0], get_wordnet_pos(word_tag[1]))
        else:  # if the post tagger did NOT annotate the given word, lemmatize the word WITHOUT POS tag
            lemma = lemmatizer.lemmatize(word_tag[0])
        lemma_text.append(lemma)
    return lemma_text


# function for stemming
def stemming(tokenized_text):
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")
    stemmed_text = []
    for word in tokenized_text:
        stem = stemmer.stem(word)
        stemmed_text.append(stem)
    return stemmed_text


# function to keep only alpharethmetic values
def only_alpha(tokenized_text):
    text_alpha = []
    for word in tokenized_text:
        word_alpha = re.sub('[^a-z A-Z]+', ' ', word)
        text_alpha.append(word_alpha)
    return text_alpha


# Method to clean tweets and instagram posts
def clean_text(text):
    # remove entities and links
    text = strip_all_entities(strip_links(text))

    # remove rt and via in case of tweet data
    text = text.lower()
    text = re.sub(r"rt", "", text)
    text = re.sub(r"via", "", text)

    # remove repost in case of instagram data
    text = re.sub(r"repost", "", text)

    # replace consecutive non-ASCII characters with a space
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # remove emojis from text
    text = emoji_pattern.sub(r'', text)

    # substitute contractions with full words
    text = replace_contractions(text)

    # tokenize text
    tokenized_text = word_tokenize(text)

    # remove all non alpharethmetic values
    tokenized_text = only_alpha(tokenized_text)

    # print("tokenized_text", tokenized_text)

    filtered_text = []
    # looping through conditions
    for word in tokenized_text:
        # check tokens against stop words, emoticons and punctuations
        if (
                word not in stop_words and word not in emoticons and word not in string.punctuation and not word.isspace() and len(
            word) > 1) or word in whitelist:
            # print("word", word)
            filtered_text.append(word)

    # lemmatize / stem words
    # tokenized_text = lemmatizing(filtered_text)
    # text = stemming(filtered_text)

    # print("tokenized_text 2", tokenized_text)

    return filtered_text  # ' '.join(tokenized_text)
