# behavioral patterns and user profiling
import spacy
from nltk.corpus import names
import pandas as pd
import operator
from collections import Counter
import task_2.preprocessing
from connect_mongo import read_mongo
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pandas import json_normalize

# Plotly imports - HTML plots
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import nltk
nltk.download('names')

from tqdm import tqdm
tqdm.pandas()


# ======================================================================================================================

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=False, use_spell_corrector=False, only_verbs_nouns=False)


# Read Twitter data
# description: bio
# original author: username
# screen_name: full user name
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1})
data = data.sample(n=1000, random_state=42)
pd.set_option('display.max_columns', None)

# get the nested fields screen_name, description from field user
nested_data = json_normalize(data['user'])
print(nested_data['description'])

nested_data['description'] = nested_data['description'].replace([None], [''])  # replace none values with empty strings

# clean text using preprocessing.py (clean_Text function)
nested_data['clean_text'] = nested_data.description.progress_map(reading.clean_text)

print(nested_data['description'])

'''
# Read Instagram data
# data = pd.read_csv("../dataset/test_cleaned.csv", index_col=False)

# clean text using preprocessing.py (clean_Text function)
data['clean_text'] = data.caption.progress_map(reading.clean_text)
'''

print(nested_data.shape)
print(data.shape)


print(nested_data)  # use to clean non-english posts
print(data)

# ======================================================================================================================


# dictionary of names and their corresponding gender
labeled_names = ([(name.lower(), 'male') for name in names.words('male.txt')] +
                 [(name.lower(), 'female') for name in names.words('female.txt')])


ner = spacy.load('en_core_web_sm')  # en_core_web_sm.load()
# Function to get the named entities from the text, then manual selection of entities to remove from description (user bio)
def get_person(text):
    test = ner(text)  # identify Named Entities

    named_ent = ''
    for X in test.ents:
        if X.label_ == 'PERSON':
            named_ent += X.text.lower() + ' '

    return named_ent.strip()


# Get list of all named entities in posts
named_entities = []
for bio in tqdm(nested_data['description']):
    named_entities.append(get_person(bio))
print("named_entities: ", named_entities)


print(data['original author'])
print(nested_data['screen_name'])



# detect the gender based on lexicons (matching is performed on lower case names)
def gender_identifier(names_list):
    all_gender_list = []  # save the gender for all the users
    # detect if a name in the dictionary matches the name in the original author field
    for name in names_list:
        longest_name_match = []
        for name_gender in labeled_names:
            if name_gender[0] in name:
                longest_name_match.append((name_gender[1], len(name_gender[0])))  # save the gender and the number of common letters
                #print(name_gender[0], ' = ', name, name_gender[1])

        #print("HERE: ", longest_name_match)
        if longest_name_match:  # check if the username does NOT match to any names
            if len(longest_name_match) == 1:  # if we have just one name match, save the gender
                all_gender_list.append(longest_name_match[0][0])  # select the gender of the largest common name
                #print("ONE ITEM - GENDER MATCH", longest_name_match[0][0])
            else:  # if there are more than 1 matching name in the list
                # sort list by common substring to get the element that has the largest common substring in the first index (0)
                longest_name_match.sort(key=lambda x: x[1], reverse=True)  # sort in descending order
                #print("IN-HERE: ", longest_name_match)

                # check the matching names if they agree on the gender, if not then we do not know the gender
                for j in range(1, len(longest_name_match)):  # get all elements except the first, as it used to check the others
                    # if the first name is the longest or has the same gender as the others that have equal length
                    if longest_name_match[0][1] == longest_name_match[j][1]:  # check if the length is the same
                        if not longest_name_match[0][0] == longest_name_match[j][0]:  # if the gender is not the same
                            # if more than two top names have the same length but different gender, the gender is unknown
                            all_gender_list.append(None)
                            #print("GENDER NOT MATCH")
                            break  # break inner loop, as there is no reason to check all items when we have gender miss-match among them
                    else:
                        # the first name is the longest or the top names, that have equal length, have the same gender
                        all_gender_list.append(longest_name_match[0][0])  # select the gender of the largest common name
                        #print("GENDER MATCH", longest_name_match[0][0])
                        break  # break inner loop, as there is no reason to check all items when we have length miss-match among them
                    if j == len(longest_name_match)-1:  # if reached the last iteration, then all the names have equal length and the same gender
                        all_gender_list.append(longest_name_match[0][0])  # select the gender of the largest common name
                        #print("LENGTH & GENDER MATCH", longest_name_match[0][0])
        else:  # no name matched the username, thus we do not know the gender
            all_gender_list.append(None)
            #print("EMPTY LIST")

    return all_gender_list


original_author_gender_list = gender_identifier(data['original author'].str.lower())
screen_name_gender_list = gender_identifier(nested_data['screen_name'].str.lower())
name_gender_list = gender_identifier(nested_data['name'].str.lower())  # check the name entities found from bio for names

named_entities_gender_list = gender_identifier(named_entities)  # check the name entities found from bio for names

print("original_author_gender_list: ", original_author_gender_list)
print("original_author_gender_list: ", len(original_author_gender_list))
print("screen_name_gender_list: ", screen_name_gender_list)
print("screen_name_gender_list: ", len(screen_name_gender_list))
print("name_gender_list: ", name_gender_list)
print("name_gender_list: ", len(name_gender_list))

print("named_entities_gender_list: ", named_entities_gender_list)
print("named_entities_gender_list: ", len(named_entities_gender_list))

final_gender_list = []
for i in range(len(original_author_gender_list)):
    # give WEIGHT to name_gender_list by counting the gender as 2 occurrences (instead of 1), as this field is the name given by the user itself
    gender_counts = Counter([original_author_gender_list[i], screen_name_gender_list[i], name_gender_list[i], name_gender_list[i]])
    gender_counts = sorted(gender_counts.items(), key=operator.itemgetter(1), reverse=True)
    print("gender_counts: ", gender_counts)

    if gender_counts[0][1] == 4: # example [('male', 4)]
        final_gender_list.append(gender_counts[0][0])
    elif gender_counts[0][1] == 3:  # example [('male', 3), ('female', 1)]
        if gender_counts[0][0] is not None:  # example [('male', 3), (None, 1)]
            final_gender_list.append(gender_counts[0][0])
        else:  # example [(None, 3), ('male', 1)]
            final_gender_list.append(gender_counts[1][0])
    elif gender_counts[0][1] == 2:  # example [('male', 2), ('female', 2)]
        if gender_counts[0][0] is not None:  # the first element is not None
            if gender_counts[1][1] == 2:  # example [('male', 2), ('female', 2)] or [('male', 2), (None, 2)]
                if gender_counts[1][0] is not None:  # example [('male', 2), ('female', 2)]
                    final_gender_list.append(None)
                else:  # example [('male', 2), (None, 2)]
                    final_gender_list.append(gender_counts[0][0])
            else:  # example [('male', 2), ('female', 1), (None, 1)]
                final_gender_list.append(gender_counts[0][0])
        else:  # the first element is None
            if gender_counts[1][1] == 2:  # example [(None, 2), ('male', 2)]
                final_gender_list.append(gender_counts[1][0])
            else:  # example [(None, 2), ('male', 1), ('female', 1)]
                final_gender_list.append(None)

    '''
    if gender_counts[0][1] >= 2:
        if not gender_counts[0][1] is None:  # if we do not have two or more None values
            final_gender_list.append(gender_counts[0][0])
        elif gender_counts[0][1] < 3:  # if not all values are None (two values are None and the third is gender)
            final_gender_list.append(gender_counts[1][0])  # the second field of Counter has the value of gender
        else:  # if all three values are None (no info about gender)
            final_gender_list.append(None)
    else:  # if we do not have at least 2 matching gender fields (i.e [male, female, None])
        final_gender_list.append(None)
    '''
print("final_gender_list: ", final_gender_list)




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

count_per_gender = Counter(final_gender_list)

# change the name of None to Unknown in order to plot the number of instances that we do not have information about the gender
count_per_gender["Unknown"] = count_per_gender.pop(None)

print("count_per_gender: ", count_per_gender)
genders = [gend_keys for gend_keys in count_per_gender.keys()]
print("genders: ", genders)
gender_occurences = [gend_values for gend_values in count_per_gender.values()]
print("gender_occurences: ", gender_occurences)

data = [go.Bar(
    x=genders,
    y=gender_occurences,
    marker=dict(colorscale='Jet', color=gender_occurences),
    text='Text posts attributed to Topic'
)]

layout = go.Layout(
    title='Gender distribution'
)

fig = go.Figure(data=data, layout=layout)

py.plot(fig, filename='gender-distribution.html')
