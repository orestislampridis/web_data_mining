import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import *
from sklearn.metrics import accuracy_score, f1_score ,precision_score, recall_score
import collections

from tqdm import tqdm
tqdm.pandas()

#load age_tweets
data = pd.read_csv(r"genders.csv",
                   encoding="utf8",
                   sep=' ',
                   error_bad_lines=False,
                   names=['_id','gender','text','description'])
pd.set_option('display.max_columns', None)
data['description']=data['description'].replace(np.nan, '', regex=True)
print(data.gender.value_counts())
gender = {'male': 1,'female': 0}
data.gender = [gender[item] for item in data.gender]

data["content"] = data["text"] + data["description"]


def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/|@]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

data["content"] = data["content"].str.lower()
data["content"] = data["content"].apply(cleanPunc)

print(data.head(3))

vectorizer = TfidfVectorizer(stop_words='english',max_features=10000, min_df = 0.01, max_df = 0.90, ngram_range=(1,4))
vectors=vectorizer.fit_transform(data["content"])
pickle.dump(vectorizer.vocabulary_,open("tfidf_gender.pkl","wb")) #save tfidf vector

X=vectors
y=data['gender']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=12)

names = [
         "Nearest Neighbors",
        "Linear SVC",
         "RBF SVM",
         "Decision Tree",
         "Random Forest",
         "AdaBoost"]

classifiers = [
    KNeighborsClassifier(12),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=3, n_estimators=20, max_features=8),
    AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                        n_estimators=800,
                        learning_rate=1,
                        algorithm="SAMME")
    ]
# k=12
# name = "Nearest Neighbors"
# clf = KNeighborsClassifier(k)

best=0
for clf,name in zip(classifiers,names):
    print("\nClassifier:",name)
    print("Fitting...")
    clf.fit(X_train, y_train)
    print("Predicting...")
    y_pred=clf.predict(X_test)
                                                                # Adaboost
    print("Accuracy:",accuracy_score(y_test, y_pred))           # Accuracy: 0.632
    print("Precission:",precision_score(y_test, y_pred))        # Precission: 0.6164383561643836
    print("Recal:",recall_score(y_test, y_pred))                # Recal: 0.7142857142857143
    print("f1_score:",f1_score(y_test, y_pred))                 # f1_score: 0.661764705882353
    print("\n")

    if best<accuracy_score(y_test, y_pred):
        best_i=name
        best=accuracy_score(y_test, y_pred)
        best_clf=clf

print(best_i)
print(best)
clf = best_clf

filename = 'clf_final_gender.sav'
pickle.dump(clf, open(filename, 'wb'))

# print(collections.Counter(y_pred)) # Counter({1: 97, 0: 28})
# print(collections.Counter(y_test)) # Counter({1: 68, 0: 57})
