import numpy as np
import pandas as pd
import seaborn as sns
import pyLDAvis
import pyLDAvis.gensim
from gensim import models
from gensim import corpora
import matplotlib.pyplot as plt
from collections import Counter
from gensim.models import Phrases
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import task_2.preprocessing
from twitter.connect_mongo import read_mongo

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=False, use_spell_corrector=True)


# Read Twitter data
data = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1})
data = data.sample(n=1000, random_state=42)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.text.progress_map(reading.clean_text)

'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

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
# LDA
# ======================================================================================================================

# Prepare bi-grams and tri-grams
list_of_list_of_tokens = data['clean_text'].tolist()

print(list_of_list_of_tokens)

# threshold: Represent a score threshold for forming the phrases -- bi-grams/tri-grams (higher means fewer phrases, default: 10.0)
bigram_model = Phrases(list_of_list_of_tokens, threshold=5.0)  # get ready bi-gram identifier
# min_count: Ignore all words and bigrams with total collected count lower than this value (default=5)
trigram_model = Phrases(bigram_model[list_of_list_of_tokens], min_count=1, threshold=5.0)  # get ready tri-gram identifier
# get possible bi-grams and tri-grams, depending on the number of times that bi-grams and tri-grams appear together
list_of_list_of_tokens = list(trigram_model[bigram_model[list_of_list_of_tokens]])


print("tokens with uni-grams, bi-grams and tri-grams combined: ", list_of_list_of_tokens)


# Prepare objects for LDA gensim implementation
dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
print("dictionary_LDA", dictionary_LDA)

# no_below: Filter words that appear in less than 3 posts
# no_above: more than 0.5 documents (fraction of total corpus size, not absolute number)
# keep_n: keep only the first 100000 most frequent tokens
dictionary_LDA.filter_extremes(no_below=3, no_above=0.5, keep_n=100000)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
print("corpus", corpus)

# Run LDA
np.random.seed(123456)
num_topics = 5
# num_topics: the number of topics
# eta: the [distribution of the] number of words per topic
# alpha: the [distribution of the] number of topics per document
lda_model = models.LdaModel(corpus, num_topics=num_topics,
                            id2word=dictionary_LDA,
                            passes=20,
                            alpha=[0.01]*num_topics,
                            eta=[0.01]*len(dictionary_LDA.keys()))

# print detected topics
for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=5):
    print(str(i) + ": " + topic)
    print()


# Allocating topics to documents

# print the document
print("document: ", data.clean_text.loc[0][:500])
# print the % of topics a document is about
print("% of topics document is about: ", lda_model[corpus[0]])  # corpus[0] means the first document.


# ======================================================================================================================
# Data Exploration and Plots
# ======================================================================================================================

topics = [lda_model[corpus[i]] for i in range(len(data))]

def topics_document_to_dataframe(topics_document, num_topics):
    res = pd.DataFrame(columns=range(num_topics))
    for topic_weight in topics_document:
        res.loc[0, topic_weight[0]] = topic_weight[1]
    return res

# Create a matrix of topic weighting (similar to TF-IDF), with documents as rows and topics as columns
document_topic = \
pd.concat([topics_document_to_dataframe(topics_document, num_topics=num_topics) for topics_document in topics]) \
  .reset_index(drop=True).fillna(0)

print("document_topic", document_topic)

# Which documents are about topic 1
print(document_topic.sort_values(1, ascending=False)[1])

print(data.clean_text.loc[91][:1000])

# ======================================================================================================================

#Looking at the distribution of topics in all documents

sns.set(rc={'figure.figsize': (10, 20)})
sns.heatmap(document_topic.loc[document_topic.idxmax(axis=1).sort_values().index])
plt.show()

# ======================================================================================================================

sns.set(rc={'figure.figsize': (10, 5)})
document_topic.idxmax(axis=1).value_counts().plot.bar(color='lightblue')
plt.show()

# ======================================================================================================================
# Wordcloud of Top N words in each topic

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = lda_model.show_topics(formatted=False)

fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)  # set the number of plots

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()

# ======================================================================================================================
# Word Count and Importance of Topic Keywords

data_flat = [w for w_list in list_of_list_of_tokens for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True, dpi=100)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030)
    ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
plt.show()

# ======================================================================================================================

#Visualizing topics

# https://cran.r-project.org/web/packages/LDAvis/vignettes/details.pdf

# size of bubble: proportional to the proportions of the topics across the N total tokens in the corpus
# red bars: estimated number of times a given term was generated by a given topic
# blue bars: overall frequency of each term in the corpus

# -- Relevance of words is computed with a parameter lambda
# -- Lambda optimal value ~0.6 (https://nlp.stanford.edu/events/illvi2014/papers/sievert-illvi2014.pdf)
vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
pyLDAvis.show(vis)