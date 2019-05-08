import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle

ps = PorterStemmer()

def stem(text_array):
    arr = []
    for t in text_array:
        stemmed = ps.stem(t)
        if not stemmed in arr:
            arr.append(stemmed)
    return arr

def removeStopword(text_array):
    return filter(lambda x: x not in set(stopwords.words('english')), text_array)

dataset = pd.read_csv(r'./emails.csv')
dataset.columns #Index(['text', 'spam'], dtype='object')
dataset.shape  #(5728, 2)

#Checking for duplicates and removing them
dataset.drop_duplicates(inplace = True)
dataset.shape  #(5695, 2)

# TRAINING
#Every mail starts with 'Subject :' will remove this from each text 
dataset['text']=dataset['text'].map(lambda text: text[9:])
# Replace all character except alphabet and number to space character
dataset['text'] = dataset['text'].map(lambda text:re.sub('[^a-zA-Z0-9]+', ' ', text)).apply(lambda x: (x.lower()).split())

def trainAndTest(is_stemming, is_remove_stopword, algorithm, ratio):

    if is_remove_stopword:
        print(111111111111111111)
        dataset['text'] = dataset['text'].map(lambda text: removeStopword(text))
    else:
        print(222222222222222222)

    if is_stemming:
        dataset['text'] = dataset['text'].map(lambda text: stem(text))
    
    

    corpus = dataset['text'].apply(lambda text_list:' '.join(list(text_list)))

    # Creating the Bag of Words model
    cv = CountVectorizer()
    X = cv.fit_transform(corpus.values)
    y = dataset.iloc[:, 1].values
    
    # Fitting Naive Bayes classifier to the Training set
    if algorithm == 'multinomial':
        classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
    elif algorithm == 'bernoulli':
        classifier = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=False, class_prior=None)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ratio, random_state=42)

    classifier.fit(X_train , y_train)

    # save model for testing
    pickle.dump(classifier, open('./learned/model.sav', 'wb'))

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
    print(accuracies.mean()) #0.9888085218938609
    
    # print(accuracy_score(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred,normalize=False)) 
    return {'accuracy': accuracy_score(y_test, y_pred)}


def testing():
    string = 'The script above uses CountVectorizer class from the sklearn.feature_extraction.text library. There are some important parameters that are required to be passed to the constructor of the class. The first parameter is the max_features parameter, which is set to 1500. This is because when you convert words to numbers using the bag of words approach, all the unique words in all the documents are converted into features'
    model = pickle.load(open('./learned/model.sav', 'rb'))
    cv = CountVectorizer()
    X = cv.fit_transform(np.array([string]))
    print(type(np.array([string])))

    predict = model.predict(X)
    print(predict)
    return predict

