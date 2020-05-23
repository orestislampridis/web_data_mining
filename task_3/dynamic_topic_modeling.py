import csv
import datetime
import logging

import gensim
import pandas as pd
import pyLDAvis
import spacy
from gensim import corpora
from gensim.models.wrappers.dtmmodel import DtmModel
from tqdm import tqdm

import preprocessing
from connect_mongo import read_mongo
from simple_preprocessing import clean_text

tqdm.pandas()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("test")

# path to the DTM binary executable file
dtm_path = r"dtm_bin/dtm-win64.exe"


class DTMcorpus(corpora.textcorpus.TextCorpus):
    def get_texts(self):
        return self.input

    def __len__(self):
        return len(self.input)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# lemmatization: achieve the root forms
def lemmatization(texts, allowed_postags=None):
    """https://spacy.io/api/annotation"""
    if allowed_postags is None:
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        # token.lemma_: root of token; token.pos_: The simple part-of-speech tag ('NOUN', 'ADJ', 'VERB', 'ADV')
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# Get our initial df with columns 'text' and 'created_at' which contains the date of the tweet's creation
df = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1, 'created_at': 1}).head(81835)

reading = preprocessing.preprocessing(convert_lower=True, use_spell_corrector=True)

df["datetime"] = pd.to_datetime(df["created_at"], errors='ignore', utc=True)

df['filtered_text'] = df.text.apply(clean_text)
# df['filtered_text'] = df.text.progress_map(lambda x: reading.clean_text(x))

data = df.filtered_text.values.tolist()

print(data[:1])

# Build the bigram and trigram models that help detect common phrases
# e.g. multi-word expressions / word n-grams – from a stream of sentences
bigram = gensim.models.Phrases(data, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data], threshold=100)

# get a sentence clubbed as a trigram or bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data[0]]])

# Form Bigrams
data_words_bigrams = make_bigrams(data)

# Initialize spacy 'de' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm',
                 disable=['parser', 'ner'])  # parser:Dependency Parsing;  ner: Named Entity Recognition
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

corpus = DTMcorpus(data_lemmatized)

# Setting the time_slices. Each time slice represents a different day
# starting from 5th of April and ending at 14th of April
t1 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 5))].shape[0]
t2 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 6))].shape[0]
t3 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 7))].shape[0]
t4 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 8))].shape[0]
t5 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 9))].shape[0]
t6 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 10))].shape[0]
t7 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 11))].shape[0]
t8 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 12))].shape[0]
t9 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 13))].shape[0]
t10 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 14))].shape[0]
# t11 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 4, 22))].shape[0]
# t12 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 5, 4))].shape[0]
# t13 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 5, 5))].shape[0]
# t14 = df.loc[(df["datetime"].dt.date == datetime.date(2020, 5, 12))].shape[0]

print(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10)
time_slices = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]

# Run Dynamic topic modeling
model = DtmModel(dtm_path, corpus, time_slices, num_topics=10,
                 id2word=corpus.dictionary)

# Topic Evolution
num_topics = 10

model.show_topics()

# Or print output
for topic_no in range(num_topics):
    print("\nTopic", str(topic_no))
    for time in range(len(time_slices)):
        print("Time slice", str(time))
        print(model.show_topic(topic_no, time, topn=10))

# Save output to file for later use
with open('output2.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    for topic_no in range(num_topics):
        csvwriter.writerow(str(topic_no))
        for time in range(len(time_slices)):
            csvwriter.writerow(str(time))
            csvwriter.writerow(model.show_topic(topic_no, time, topn=10))

doc_topic, topic_term, doc_lengths, term_frequency, vocab = model.dtm_vis(time=0, corpus=corpus)
vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
                               vocab=vocab, term_frequency=term_frequency)

# pyLDAvis.display(vis_wrapper)
pyLDAvis.save_html(vis_wrapper, 'dtp.html')
