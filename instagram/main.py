import json
import pandas as pd
from datetime import datetime

# reading the JSON data using json.load()
file = 'test.json'

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

column_list = set(column_list)
print(column_list)
print(len(column_list))

keys_list = set(keys_list)
print(keys_list)
print(len(keys_list))

test = pd.DataFrame(tweets)
print(test)

clist = list(test.columns)  # list of all column names
print(clist)
print(len(clist))  # number of all columns

# save cleaned data, from dataframe, to csv
test.to_csv('test_cleaned.csv', index=False, encoding='utf-8')

data = pd.read_csv("test_cleaned.csv", index_col=False)

print("NEW CSV:\n", data["date"])
