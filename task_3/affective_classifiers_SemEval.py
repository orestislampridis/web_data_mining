import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import task_2.preprocessing
import nltk
nltk.download('averaged_perceptron_tagger')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from tqdm import tqdm
tqdm.pandas()

#load SemEval emotions classification
file="../dataset/task3-affect_recognision/2018-11 emotions-classification-train.txt"
SemEval = pd.read_csv(file, sep=" ",delimiter="\t", encoding="utf8")
#keep only what we need
SemEval=SemEval[['ID','Tweet','anger','joy','disgust','fear','sadness','surprise']]
emotion_categories = list(SemEval.columns.values)
emotion_categories = emotion_categories[2:]
#Check SemEval
print("Number of data in SemEval emotions dataset =",SemEval.shape[0])
print("Number of emotion categories =",len(emotion_categories))
print("Emotion categories =", ', '.join(emotion_categories))
print("\n")
print("**Sample emotion data:**")
SemEval.head()

plt.figure(figsize=(15,8))
sums=SemEval.iloc[:,2:].sum().values
ax= sns.barplot(emotion_categories,sums )
plt.title("Posts/Tweets in each category", fontsize=22)
plt.ylabel('Number of posts/tweets', fontsize=15)
plt.xlabel('Post/tweet Type ', fontsize=15)
plt.show()

rowsums = SemEval.iloc[:,2:].sum(axis=1).value_counts()
plt.figure(figsize=(15,8))
ax = sns.barplot(rowsums.index, rowsums.values)
plt.title("Categories per post/tweet", fontsize=22)
plt.ylabel('Number of Occurrences', fontsize=15)
plt.xlabel('Number of categories', fontsize=15)
plt.show()

# create object of class preprocessing to clean data
reading = task_2.preprocessing.preprocessing(convert_lower=False, use_spell_corrector=True)
SemEval['Tweet'] = SemEval.Tweet.progress_map(reading.clean_text)


#print(SemEval['Tweet'].iloc[:2])
def list2string(list):
        return ','.join(map(str, list))

SemEval['Tweet'] = [list2string(list) for list in SemEval['Tweet']]
#print(SemEval['Tweet'].iloc[:2])

#spliting data
train, test = train_test_split(SemEval, random_state=7, test_size=0.20)

x_train=train['Tweet']
x_test=test['Tweet']

y_train = train.drop(labels = ['ID','Tweet'], axis=1)
y_test = test.drop(labels = ['ID','Tweet'], axis=1 )


#OneVsRest approach
#Linear SVC
SVC_OneVsRest = Pipeline([
                ('tfidf', TfidfVectorizer(encoding='utf-8',ngram_range=(1,3))),
                ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
            ])
i = 1
pred_onevsrest_SVC = []
print("---OneVsRest approach with linear SVC---")
for category in emotion_categories:
    print('%d)%s tweets' %(i,category))
    SVC_OneVsRest.fit(x_train, train[category])

    filename = 'Task 3 Classifiers\OneVsRest_SVC_{}'.format(category)
    pickle.dump(SVC_OneVsRest, open(filename, 'wb'))

    pred = SVC_OneVsRest.predict(x_test)
    pred_onevsrest_SVC.append(pred)
    print('   Test accuracy: {}'.format(accuracy_score(test[category], pred)))
    i=i+1

#OneVsRest approach
#LogisticRegression
Log_OneVsRest = Pipeline([
    ('tfidf', TfidfVectorizer(encoding='utf-8', ngram_range=(1, 3))),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)),
])
i = 1
pred_onevsrest_Log = []
print("---OneVsRest approach with Logistic Regression---")
for category in emotion_categories:
    print('%d)%s tweets' % (i, category))
    Log_OneVsRest.fit(x_train, train[category])

    filename = 'Task 3 Classifiers\OneVsRest_Log_{}'.format(category)
    pickle.dump(Log_OneVsRest, open(filename, 'wb'))

    pred = Log_OneVsRest.predict(x_test)
    pred_onevsrest_Log.append(pred)
    print('   Test accuracy: {}'.format(accuracy_score(test[category], pred)))
    i = i + 1

#Binary Relevance approach
print("---Binary Relevance approach---")
SVC_BinaryRel = Pipeline([
                ('tfidf', TfidfVectorizer(encoding='utf-8',ngram_range=(1,3))),
                ('clf', BinaryRelevance(LinearSVC())),
            ])

SVC_BinaryRel.fit(x_train, y_train)

filename = 'Task 3 Classifiers\Binary_Relevance_SVC'
pickle.dump(SVC_BinaryRel, open(filename, 'wb'))

predictions_binary = SVC_BinaryRel.predict(x_test)

print('Test accuracy: {}'.format(accuracy_score(y_test,predictions_binary)))



# Classifier Chains approach
print("---Classifier Chains approach---")
SVC_Chains = Pipeline([
                ('tfidf', TfidfVectorizer(encoding='utf-8',ngram_range=(1,3))),
                ('clf', ClassifierChain(LinearSVC())),
            ])

SVC_Chains.fit(x_train, y_train)

filename = 'Task 3 Classifiers\Classifier_Chains_SVC'
pickle.dump(SVC_Chains, open(filename, 'wb'))

predictions_chains = SVC_Chains.predict(x_test)
print('Test accuracy: {}'.format(accuracy_score(y_test,predictions_chains)))



#Label Powerset approach
print("---Label Powerset approach---")
SVC_LabelPow = Pipeline([
                ('tfidf', TfidfVectorizer(encoding='utf-8',ngram_range=(1,3))),
                ('clf', ClassifierChain(LinearSVC())),
            ])
SVC_LabelPow.fit(x_train, y_train)

filename = 'Task 3 Classifiers\Label_Powerset_SVC'
pickle.dump(SVC_Chains, open(filename, 'wb'))

predictions_label = SVC_LabelPow.predict(x_test)
print('Test accuracy: {}'.format(accuracy_score(y_test,predictions_label)))