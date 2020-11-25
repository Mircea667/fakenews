# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:35:25 2020

@author: PORTATIL
"""
import os
import glob
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from nltk.corpus import stopwords
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle


# Cargar CSV

df = pd.read_csv(r"C:\Users\PORTATIL\Desktop\TFG\dataset_tfg\isot_dataset\combinado.csv")

#Convert NaN values to empty string
nan_value = float("NaN")

df.replace("", nan_value, inplace=True)

df.dropna( inplace=True)

print((df.columns))


# Definición de los datos

X= df['title'].astype(str) + ' ' + df['body'].astype(str)
y = df['Category']
W1=X
z=y

#y_list=y.tolist()

# Dividir los datos 70% train 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70,test_size=0.30)

#Aplicar TFIDF.
tfIdfVectorizer=TfidfVectorizer(smooth_idf=True,max_features=1000,use_idf=True)
#tfIdfVectorizer = TfidfVectorizer()



X_vectorizer = tfIdfVectorizer.fit_transform(X_train)
X_vector=tfIdfVectorizer.transform(X_test)

X_train2=X_vectorizer.toarray()
X_test2=X_vector.toarray()

W_vectorizer=tfIdfVectorizer.fit_transform(W1)
W_learning=W_vectorizer.toarray()

W = coo_matrix(W_learning)

W,z=shuffle(W,z)

#ENTRENAMIENTO NAIVE BAYES
# Definir modelo de clasificación, Naive Bayes, Decision Tree y Random Forest. 
naive_bayes_classifier = BernoulliNB()
naive_bayes_classifier=naive_bayes_classifier.fit(X_train2, y_train)

pred_NB = naive_bayes_classifier.predict(X_test2)

#ENTRENAMIENTO DECISION TREE.
decision_tree_classifier=DecisionTreeClassifier()
decision_tree_classifier=decision_tree_classifier.fit(X_train2, y_train)

pred_DT=decision_tree_classifier.predict(X_test2)


#tree.plot_tree(decision_tree) 

#ENTRENAMIENTO RANDOM FOREST
random_forest=RandomForestClassifier(n_estimators=3)
random_forest=random_forest.fit(X_train2,y_train)

pred_RF=random_forest.predict(X_test2)

# compute the performance measures
score1 = metrics.accuracy_score(y_test,pred_NB )
score2 = metrics.accuracy_score(y_test,pred_DT)
score3 = metrics.accuracy_score(y_test,pred_RF)

##LEARNING CURVES
#NAIVE BAYES
train_sizes, train_scores, test_scores = learning_curve(BernoulliNB(), W, z, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes, train_mean, '--', color="m",  label="Training score")
plt.plot(train_sizes, test_mean, color="y", label="Cross-validation score")

plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('NB')
plt.show()


#DECISION TREE
train_sizes2, train_scores2, test_scores2 = learning_curve(DecisionTreeClassifier(), W, z, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

train_mean2 = np.mean(train_scores2, axis=1)
train_std2 = np.std(train_scores2, axis=1)

test_mean2 = np.mean(test_scores2, axis=1)
test_std2 = np.std(test_scores2, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes2, train_mean2, '--', color="m",  label="Training score")
plt.plot(train_sizes2, test_mean2, color="y", label="Cross-validation score")

plt.fill_between(train_sizes2, train_mean2 - train_std2, train_mean2 + train_std2, color="#DDDDDD")
plt.fill_between(train_sizes2, test_mean2 - test_std2, test_mean2 + test_std2, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('DT')
plt.show()


#RANDOM FOREST
train_sizes3, train_scores3, test_scores3 = learning_curve(RandomForestClassifier(), W, z, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))

train_mean3 = np.mean(train_scores3, axis=1)
train_std3 = np.std(train_scores3, axis=1)

test_mean3 = np.mean(test_scores3, axis=1)
test_std3 = np.std(test_scores3, axis=1)

plt.subplots(1, figsize=(10,10))
plt.plot(train_sizes3, train_mean3, '--', color="m",  label="Training score")
plt.plot(train_sizes3, test_mean3, color="y", label="Cross-validation score")

plt.fill_between(train_sizes3, train_mean3 - train_std3, train_mean3 + train_std3, color="#DDDDDD")
plt.fill_between(train_sizes3, test_mean3 - test_std3, test_mean3 + test_std3, color="#DDDDDD")

plt.title("Learning Curve")
plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
plt.tight_layout()
plt.savefig('RF')
plt.show()


#MATRIZ DE CONFUSION DEL NAIVE BAYES
print("accuracy:   %0.3f" % score1)

print(metrics.classification_report(y_test, pred_NB,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred_NB))

print('------------------------------')
#MATRIZ DE CONFUSION DECISION TREE
print("accuracy:   %0.3f" % score2)

print(metrics.classification_report(y_test,pred_DT ,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test, pred_DT))

print('------------------------------')

#MATRIZ DE CONFUSION RANDOM FOREST
print("accuracy:   %0.3f" % score3)

print(metrics.classification_report(y_test, pred_RF,
                                            target_names=['Positive', 'Negative']))

print("confusion matrix:")
print(metrics.confusion_matrix(y_test,pred_RF))

print('------------------------------')


