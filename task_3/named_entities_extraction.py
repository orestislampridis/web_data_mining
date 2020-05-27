from pprint import pprint
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
import pandas as pd
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import requests
import re
import operator
import numpy as np
from nltk.tree import Tree
from nltk import ne_chunk_sents, ne_chunk, pos_tag, word_tokenize
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import task_2.preprocessing
from connect_mongo import read_mongo

from tqdm import tqdm
tqdm.pandas()

import nltk
nltk.download('words')
nltk.download('maxent_ne_chunker')


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
data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

print(data.shape)

# drop the rows that contain empty captions
# inplace=True: modify the DataFrame in place (do not create a new object) - returns None
# data[data['clean_text'].str.len() < 1]  # alternative way
data.drop(data[data['clean_text'].progress_map(lambda d: len(d)) < 1].index, inplace=True)  # drop the rows that contain empty captions
data.reset_index(drop=True, inplace=True)  # reset index needed for dataframe access with indices

# ======================================================================================================================
# Time Series
'''
data.date = pd.to_datetime(data.date)
data.set_index('date', inplace=True)

data["likes"].plot(figsize=(20, 10), linewidth=5, fontsize=20)
# data.xlabel('Year', fontsize=20)  # specify the x-axis to be YEARS, but we want less than months
plt.show()
'''
# ======================================================================================================================

print("data: ", data)  # use to clean non-english posts

words_of_posts = data['clean_text'].tolist()

# ======================================================================================================================


'''
pprint([(X.text, X.label_) for X in doc.ents])

pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])




# print count per NER label
labels = [x.label_ for x in sentence.ents]
Counter(labels)

#  three most frequent tokens
items = [x.text for x in sentence.ents]
Counter(items).most_common(3)



# lemmatize
print([(x.orth_, x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[20]))]])
'''


# ======================================================================================================================
# Remove most common named entities (NER)
# ======================================================================================================================

def get_continuous_chunks(named_ent, text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))  # identify Named Entities
    continuous_chunk = []
    current_chunk = []
    for chunk in chunked:
        if type(chunk) == Tree:
            current_chunk.append(" ".join([token for token, pos in chunk.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    named_ent += continuous_chunk

    return named_ent


ner = spacy.load('en_core_web_sm')  # en_core_web_sm.load()
# Function to get the named entities from the text, then manual selection of entities to remove from clean_text (posts)
def get_entities(text):
    text = " ".join(text)
    test = ner(text)  # identify Named Entities

    #pprint([(X.text, X.label_) for X in test.ents])
    #pprint([(X, X.ent_iob_, X.ent_type_) for X in test])

    named_ent = [X.text for X in test.ents]
#    print("RETURN", named_entities)

    return named_ent


# Get list of all named entities in posts
named_entities = []
for clean_text in tqdm(data['clean_text']):
    named_entities += get_entities(clean_text)
    #named_entities += get_continuous_chunks(named_entities, text)
print("named_entities: ", named_entities)

#named_entities = np.array(named_entities)

# Get Dictionary with Counts of named_entities
named_entities_counts = Counter(named_entities)
print("named_entities_counts 1: ", named_entities_counts)

named_entities_counts = sorted(named_entities_counts.items(), key=operator.itemgetter(1), reverse=True)

print("named_entities_counts 2: ", named_entities_counts)
print("len 2: ", len(named_entities_counts))


# ======================================================================================================================

# wordcloud of the most common entities
wordcloud_words_freq = dict()
for tupl in named_entities_counts:
    wordcloud_words_freq[tupl[0]] = tupl[1]

plt.figure(figsize=(20, 10), facecolor='k')
wc = WordCloud(width=1600, height=800, background_color="black")
wc.generate_from_frequencies(wordcloud_words_freq)
plt.title("Most common entities", fontsize=20)
plt.imshow(wc.recolor(colormap='Pastel2', random_state=17), alpha=0.98, interpolation="bilinear")
plt.axis('off')
plt.tight_layout()
plt.show()

# ======================================================================================================================

'''

# Create final list of 60% most occurring named entities to remove from text
common_entities = []
for i in np.arange(0, int(0.6 * len(named_entities_counts))):
    common_entities.append(named_entities_counts[i][0])  # [ ,named_entities_counts[i][1]]


print("common_entities: ", common_entities)
print("len common_entities: ", len(common_entities))


entities_to_remove = common_entities[: int(0.7 * len(common_entities))]  # get the first 70% of most common entities to remove
print(entities_to_remove)
print(len(entities_to_remove))

entities_to_remove = sorted(entities_to_remove)
print(entities_to_remove)
print(len(entities_to_remove))


# Function for removal
def remove_entities(post):
    for entity in entities_to_remove:
        if ' '+entity+' ' in post:
            post = post.replace(entity+' ', '')
        elif ' '+entity+'.' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+',' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+':' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+'-' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+';' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+'"' in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+"'" in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+"]" in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+")" in post:  # added later
            post = post.replace(' '+entity, '')
        elif ' '+entity+"?" in post:
            post = post.replace(' '+entity, '')
        elif ' '+entity+"!" in post:  # added later
            post = post.replace(' '+entity, '')
        elif '"'+entity+' ' in post:
            post = post.replace(entity+' ', '')
        elif "'"+entity+' ' in post:
            post = post.replace(entity+' ', '')
        elif "["+entity+' ' in post:
            post = post.replace(entity+' ', '')
        elif "("+entity+' ' in post: # added later
            post = post.replace(entity+' ', '')
        elif "["+entity+']' in post:
            post = post.replace(entity, '')
        elif "("+entity+')' in post: # added later
            post = post.replace(entity, '')
        elif "'"+entity+"'" in post:
            post = post.replace(entity, '')
        elif '"'+entity+'"' in post:
            post = post.replace(entity, '')
    return post


# Remove final named entities from clean_text
data['clean_text'] = [remove_entities(x) for x in data['clean_text']]

print(data['clean_text'])

'''
