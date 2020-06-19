
import pickle
import re

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

pd.set_option('display.max_columns', None)

# Read instagram Data
file1 = "../dataset/insta_data_cleaned.csv"
insta = pd.read_csv(file1, sep='~', encoding="utf8")
insta = insta[['_id', 'owner_username', 'caption']]
print("\nInsta posts", insta.head(3))


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

insta['caption'] = insta['caption'] .str.lower()
insta['caption'] = insta['caption'] .apply(cleanPunc)

# load tfidf
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 3),
                             vocabulary=pickle.load(open("tfidf_gender.pkl", 'rb')))
# transform description to tfidf
X = vectorizer.fit_transform(insta['caption'])

# load classifier
filename = 'clf_final_gender.sav'
clf = pickle.load(open(filename, 'rb'))

print("\nPredicting with {}...".format(clf.__class__.__name__))
y = clf.predict(X)
gender = {1: 'male', 0: 'female'}
y = [gender[item] for item in y]
#print(y)
insta['gender'] = y

# keep what we need
insta = insta[['_id','owner_username','caption', 'gender']]
print("\n Results:")
print(insta.head(3))

# Save original author and original text along with predicted age group
insta.to_csv('predicted_genders_insta.csv')

# Plot the results
counts =insta['gender'].value_counts()

plt.bar(counts.index[:], counts.values[:], color=(0.9, 0.0, 0.3, 0.8))
plt.title('Instagram users gender counts')
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.show()