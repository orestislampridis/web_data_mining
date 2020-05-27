import matplotlib.pyplot as plt
import pandas as pd
from pandas import json_normalize

from connect_mongo import read_mongo

# Read Twitter data
# description: bio
# original author: username
# screen_name: full user name
data = read_mongo(db='twitter_db', collection='twitter_collection',
                  query={'original author': 1, 'user': 1})
pd.set_option('display.max_columns', None)

print(data)

# get the nested fields screen_name, description from field user
nested_data = json_normalize(data['user'])

author_df = (data['original author'])
desc_df = (nested_data['description'])

concat_df = pd.concat([author_df, desc_df], axis=1, sort=False)

print(concat_df)

# Check for regular expression matching year of birth 1940-2009
year_desc_df = concat_df['original author'].str.extract(r'(19[456789]\d|20[0]\d)')
year_bio_df = concat_df['description'].str.extract(r'(19[456789]\d|20[0]\d)')

year_desc_df = year_desc_df.dropna()
year_bio_df = year_bio_df.dropna()

# Append one dataframe to the other to unify the years of birth
merged_df = year_desc_df.append(year_bio_df).sort_index()
merged_df.columns = ['year']

# Drop duplicates from above procedure
merged_df = merged_df.groupby(merged_df.index).first()
merged_df['year'] = pd.to_numeric(merged_df['year'], errors='ignore')
print(merged_df)

# Substract from current year to get age
merged_df["age"] = merged_df["year"].apply(lambda x: 2020 - x)

# Create beautiful plots
print(merged_df)
merged_df.age.plot(kind='hist')
plt.show()

merged_df.age.plot(kind='kde', xlim=(0, 100), grid=True)
plt.show()
