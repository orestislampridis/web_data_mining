import pandas as pd
import task_2.preprocessing
from twitter.connect_mongo import read_mongo
import task_3.LDA_analysis
import task_3.emerging_topics_wordcloud

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=False, use_spell_corrector=True, only_verbs_nouns=True)


# Read Twitter data
data = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'created_at': 1})
data = data.sample(n=1000, random_state=42)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.text.progress_map(reading.clean_text)

'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

data.drop(['text'], axis=1, inplace=True)

print(data.shape)


# further filter stopwords
more_stopwords = ['tag', 'not', 'let', 'wait', 'set', 'put', 'add', 'post', 'give', 'way', 'check', 'think', 'www', 'must', 'look', 'call', 'minute', 'com', 'thing', 'much', 'happen', 'hahaha']
data['clean_text'] = data['clean_text'].map(lambda clean_text: [word for word in clean_text if word not in more_stopwords])


# drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
# data[data['clean_text'].str.len() < 1]  # alternative way
data.drop(data[data['clean_text'].map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty captions
data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices


print(data)  # use to clean non-english posts


# ======================================================================================================================
# Create time series
# ======================================================================================================================

data['datetime'] = pd.to_datetime(data['created_at'])
data = data.set_index('datetime')
data.drop(['created_at'], axis=1, inplace=True)
print(data)

'''
data_per_day = []
for group in data.groupby(data.index.date):  # split dataframe into multiple dataframes with data per single day
    print(group[1])
    data_per_day.append(group[1])
'''
data_per_day = [group[1] for group in data.groupby(data.index.date)]


# ======================================================================================================================
# LDA analysis using temporal information (find topics per day)
# ======================================================================================================================

prev_word_freq = {}
emerging_topics_per_period = []
for data in tqdm(data_per_day):
    print("ALL DATES: ", set(data.index.date))
    if len(data.index) > 20:  # Check if dataframe has more than 20 posts  # data['clean_text'].empty
        word_freq = task_3.LDA_analysis.LDA_analysis(data, pyLDAvis_show=False, plot_graphs=False)
        print("word_freq: ", word_freq)

        emergingWords = []
        # the value of the word determines if a keywords is emerging by comparing the current frequency with the previous one
        wordValue = {}
        if prev_word_freq:  # if it is not the first time running LDA
            for word in word_freq:
                if word in prev_word_freq:
                    value = prev_word_freq[word] - word_freq[word]
                else:
                    value = word_freq[word]
                wordValue[word] = value

            max_val = max(wordValue.values())  # the max value of word frequencies difference
            min_val = min(wordValue.values())  # the min value of word frequencies difference

            # save the most emerging keywords
            for words in wordValue:
                scaled_wordValue = (wordValue[words] - min_val) / (max_val - min_val)  # MinMax Scaling
                if scaled_wordValue >= 0.2:
                    emergingWords.append((words, scaled_wordValue))

            print("emergingWords", emergingWords)
            emerging_topics_per_period.append((set(data.index.date), emergingWords))

        prev_word_freq = word_freq


wordcloud_words = []  # list of lists of all words that are product from the temporal analysis of the lda topics, per slice of topic
wordcloud_words_freq = dict()  # dictionary of all temporal analysis words, as keys, and their frequency as values
for elem in emerging_topics_per_period:
    print("Get only the date, instead of tuple: ", next(iter(elem[0])))
    print("emerging_topics_per_period: ", elem)

    words_per_emergin_topic = []
    for tupl in elem[1]:
        words_per_emergin_topic.append(tupl[0])
        wordcloud_words_freq[tupl[0]] = tupl[1]
    wordcloud_words.append(words_per_emergin_topic)

task_3.emerging_topics_wordcloud.emerg_topics_wordcloud(wordcloud_words, wordcloud_words_freq)
