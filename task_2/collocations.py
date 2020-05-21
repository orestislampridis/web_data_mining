"""
Script used for generating collocations

Collocations are phrases or expressions containing multiple words, that are highly likely to co-occur.
    e.g. "social network", "web data mining", "school holiday", "Pizza Hut" etc etc
"""

import nltk
import pandas as pd
from nltk.corpus import stopwords

from connect_mongo import read_mongo
from simple_preprocessing import clean_text

# Get english stopwords
en_stopwords = set(stopwords.words('english'))


# Filter out for collocations not containing stop words and filter for only the following structures:
# Bigrams: (Noun, Noun), (Adjective, Noun)
# Trigrams: (Adjective/Noun, Anything, Adjective/Noun)
# This is a common structure used in literature and generally works well.

# function to filter for bigrams
def filter_bigrams(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    acceptable_types = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    second_type = ('NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in acceptable_types and tags[1][1] in second_type:
        return True
    else:
        return False


# function to filter for trigrams
def filter_trigrams(ngram):
    if '-pron-' in ngram or 't' in ngram:
        return False
    for word in ngram:
        if word in en_stopwords or word.isspace():
            return False
    first_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    third_type = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    tags = nltk.pos_tag(ngram)
    if tags[0][1] in first_type and tags[2][1] in third_type:
        return True
    else:
        return False


# Get our initial df with text column
full_df = read_mongo(db='twitter_db', collection='twitter_collection', query={'text': 1})

df = full_df.sample(n=20000, random_state=42)
df['filtered_text'] = df.text.apply(lambda x: clean_text(x))

tokens_list = df['filtered_text'].to_list()
tokens = [item for items in tokens_list for item in items]

# Generate all possible bigrams and trigrams
bigrams = nltk.collocations.BigramAssocMeasures()
trigrams = nltk.collocations.TrigramAssocMeasures()
bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)
trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(tokens)

# Rank the most frequent bigrams and trigrams
bigram_freq = bigramFinder.ngram_fd.items()
bigramFreqTable = pd.DataFrame(list(bigram_freq), columns=['bigram', 'freq']).sort_values(by='freq', ascending=False)

trigram_freq = trigramFinder.ngram_fd.items()
trigramFreqTable = pd.DataFrame(list(trigram_freq), columns=['trigram', 'freq']).sort_values(by='freq', ascending=False)

print(bigramFreqTable)
print(trigramFreqTable)

# Remove any adjacent spaces, stop words, articles, prepositions or pronouns for bigrams and trigrams
filtered_bigramFreqTable = bigramFreqTable[bigramFreqTable.bigram.map(lambda x: filter_bigrams(x))]
filtered_trigramFreqTable = trigramFreqTable[trigramFreqTable.trigram.map(lambda x: filter_trigrams(x))]

print(filtered_bigramFreqTable)
print(filtered_trigramFreqTable)

# filter for only those with more than 50 occurences
bigramFinder.apply_freq_filter(50)
trigramFinder.apply_freq_filter(50)
bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram', 'PMI']).sort_values(
    by='PMI', ascending=False)
trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)), columns=['trigram', 'PMI']).sort_values(
    by='PMI', ascending=False)

# Save to csv
filtered_bigramFreqTable.to_csv('filtered_bigramFreqTable.csv', index=False)
filtered_trigramFreqTable.to_csv('filtered_trigramFreqTable.csv', index=False)
