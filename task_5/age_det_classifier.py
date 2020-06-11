import numpy as np
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
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
from sklearn.metrics import accuracy_score, f1_score ,precision_score, recall_score
import collections


#load age_tweets
data = pd.read_csv(r"age_tweets.csv", encoding="utf8")
#saved data has column 'Unnamed: 0' as the previous id
data.rename(columns={'Unnamed: 0':'_id'}, inplace=True)
bins = [0, 21, 36,100]
labels = ["very_young","young","young_in_heart"] #very young->[0,20], young->[21,35], young in heart->[36,100]
data['binned'] = pd.cut(data['age'], bins=bins, labels=labels)
data['description']=data['description'].replace(np.nan, '', regex=True)
pd.set_option('display.max_columns', None)
print(data.head())


# #plots data balance
# counts = data['binned'].value_counts()
# plt.bar(counts.index[:], counts.values[:], color = (0.0,0.0,1,0.5))
# plt.title('Twitter users age groups counts')
# plt.xlabel('Categories')
# plt.ylabel('Counts')
# plt.show()


# function to clean the word of any punctuation or special characters
# we need slang and emojis as they reflect the difference between age groups
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned

# function to count emojis
def emoji_count(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               "]+", flags=re.UNICODE)

    counter = 0
    datas = list(text)  # split the text into characters

    for word in datas:
        counter += len(emoji_pattern.findall(word))
    return counter

# function to slang words
def slang_count(text):
    slang_data = []
    with open("slang.txt", 'r', encoding="utf8") as exRtFile:
        exchReader = csv.reader(exRtFile, delimiter=':')

        for row in exchReader:
            slang_data.append(row[0].lower())

    counter = 0
    data = text.lower().split()
    for word in data:

        for slang in slang_data:

            if slang == word:
                counter += 1
    return counter

#count slang and emojis at text and description
data["slang_count"] = ""
data["emoji_count"] = ""
for i in range(0,len(data)):

    data["slang_count"].iloc[i] = slang_count(data['description'].iloc[i])
    data["slang_count"].iloc[i] += slang_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] = emoji_count(data['text'].iloc[i])
    data["emoji_count"].iloc[i] += emoji_count(data['description'].iloc[i])

#convert to lower and remove punctuation or special characters
data['description']=data['description'].str.lower()
data['description']=data['description'].apply(cleanPunc)

#print(data.head())

#keep the columns we need
data=data[['_id','description','emoji_count','slang_count','binned']]
print(data.head())

#convert description to tf idf vector and pickle save vectorizer
vectorizer = TfidfVectorizer(stop_words='english',max_features=10000,ngram_range=(1,3))
vectors=vectorizer.fit_transform(data['description'])
pickle.dump(vectorizer.vocabulary_,open("tfidf_age.pkl","wb")) #save tfidf vector

#save sparce vectors to pd to use with other features
vectors_pd=pd.DataFrame(vectors.toarray())

#scale emojis_count and slang_count to [0,1]
scaler = MinMaxScaler()
data[['emoji_count', 'slang_count']] = scaler.fit_transform(data[['emoji_count', 'slang_count']])

#create dataframe X, y for train
X=pd.concat([vectors_pd,data['emoji_count'],data['slang_count']],axis=1)
y=data.iloc[:,4]

# print(X)
# print(y)
#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

# uncomment to try many classifiers to find the best(adaboost)
# names = [
#          "Nearest Neighbors",
#         "Linear SVC",
#          "RBF SVM",
#          "Decision Tree",
#          "Random Forest",
#          "AdaBoost",
#          "Naive Bayes"]
#
# classifiers = [
#     KNeighborsClassifier(3),
#     SVC(kernel="linear", C=0.025),
#     SVC(gamma=2, C=1),
#     DecisionTreeClassifier(max_depth=5),
#     RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     AdaBoostClassifier(),
#     GaussianNB()]
#adaboost was the best classifier
name="AdaBoost"
clf =AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=2,
    algorithm="SAMME")
#uncomment to try several clf
#best=0
#for clf,name in zip(classifiers,names):

print("\nClassifier:",name)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))  # Accuracy: 0.720873786407767
print("Precission:",precision_score(y_test, y_pred,average='macro'))  # Precission: 0.7225022104332449
print("Recal:",recall_score(y_test, y_pred,average='macro')) # Recal: 0.7211324853708959
print("f1_score:",f1_score(y_test, y_pred,average='macro')) # f1_score: 0.7200576583813273
print("\n")

#     if best<accuracy_score(y_test, y_pred):
#         best_name=name
#         best=accuracy_score(y_test, y_pred)

# save the model to disk
filename = 'adaboost_final.sav'
pickle.dump(clf, open(filename, 'wb'))



# print(collections.Counter(y_pred)) # Counter({'young': 143, 'young in heart': 145, 'very young': 124})
# print(collections.Counter(y_test)) # Counter({'young': 151, 'young in heart': 126, 'very young': 135})