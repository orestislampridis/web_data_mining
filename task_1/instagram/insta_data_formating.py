import json
import pandas as pd
from datetime import datetime
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException


# reading the JSON data using json.load()
file = 'dataset/test.json'  # instagram data

tweets = []
for line in open(file, 'r', encoding="utf8"):
    tweets.append(json.loads(line))

keys_list = []  # collect all the keys from the nested dictionaries
column_list = []  # column list that have nested dictionaries
for x in tweets:
    for column in x:
        if isinstance(x[column], dict):  # check if a column has a dictionary as values
            column_list.append(column)  # save columns that have nested dictionaries as data
            for key, val in x[column].items():
                keys_list.append(key)  # collect all the keys from the nested dictionaries
                if key == "$numberInt" or key == "$numberLong":
                    x[column] = int(val)
                elif key == "$numberDouble":
                    x[column] = float(val)
                elif key == "$date":
                    for date_key, date_val in val.items():
                        unix_time = int(int(date_val) / 1000)  # divide with 1000 to convert ISO-date to UNIX time
                        x[column] = datetime.utcfromtimestamp(unix_time).strftime('%Y-%m-%d %H:%M:%S')  # convert UNIX time to humanly readable date
                        #print(x[column])
                elif key == "$oid":
                    x[column] = str(val)

# ======================================================================================================================

# column list containing columns that needed data type conversion
column_list = set(column_list)
print("column_list for columns that need cleaning from nested jsons: ", column_list)
print(len(column_list))

# ======================================================================================================================

# Bson (mongoDB's data types on nested jsons)
keys_list = set(keys_list)
print("Bson keys_list to remove: ", keys_list)
print(len(keys_list))

# ======================================================================================================================

# convert data structure to dataframe
df = pd.DataFrame(tweets)

# ======================================================================================================================

# columns that still need data conversion
rest_columns = list(df.columns)
for column in column_list:
    rest_columns.remove(column)
print("columns that still need data conversion: ", rest_columns)
print(len(rest_columns))

# ======================================================================================================================

# Data conversion to the rest columns (that still need data conversion)
df["caption"] = df["caption"].astype(str)
df["caption_mentions"] = df["caption_mentions"].astype(str)
df["caption_hashtags"] = df["caption_hashtags"].astype(str)
df["tagged_users"] = df["tagged_users"].astype(str)
#print(df["location"])
df["location"] = df["location"].astype(str)  # MIGHT BE COORDINATES
#print(df["_location"])
df["_location"] = df["_location"].astype(str)  # MIGHT BE COORDINATES
df["is_video"] = df["is_video"].astype(bool)
df["owner_private"] = df["owner_private"].astype(bool)
df["owner_viewable_story"] = df["owner_viewable_story"].astype(bool)
df["owner_verified"] = df["owner_verified"].astype(bool)
df["owner_username"] = df["owner_username"].astype(str)

# ======================================================================================================================

# filter all the non-english posts (with probability of being english less than 0.6)
print(df.shape)
for index, row in df['caption'].iteritems():
    try:
        languages = detect_langs(row)  # detect_langs: predict the languanges in text and their probabilities

        langs = [l.lang for l in languages]  # get a list with all languages
        langs_probas = [l.prob for l in languages]  # get a list with all language probabilities
        #print("text lang stats: ", langs, langs_probas, langs.index('en'))

        if langs_probas[langs.index('en')] < 0.9:  # if the probability of the text language being english is less than 0.9
            df = df[df.caption != row]  # drop the row
    except LangDetectException:  # in case detect finds sentence that do not contain strings. i.e. only numbers
        df = df[df.caption != row]  # drop the row
    except ValueError:  # in case the post in not in english (catch exception of langs.index('en'))
        df = df[df.caption != row]  # drop the row

print(df)
print(df.shape)

# ======================================================================================================================

clist = list(df.columns)  # list of all column names
print(clist)
print(len(clist))  # number of all columns

# ======================================================================================================================

# save cleaned data, from dataframe, to csv
df.to_csv('dataset/test_cleaned.csv', index=False, encoding='utf-8')

data = pd.read_csv("dataset/test_cleaned.csv", index_col=False)

print("NEW CSV:\n", data["date"])
