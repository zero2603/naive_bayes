import os
import sys
import collections
import re
import math
import copy
import json
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 

# Document class to store email instances easier
class Document:
    text = ""
    word_freqs = {}

    # spam or ham
    true_class = ""
    learned_class = ""

    # Constructor
    def __init__(self, text, counter, true_class):
        self.text = text
        self.word_freqs = counter
        self.true_class = true_class

    def getText(self):
        return self.text
    
    def setText(self, new_text):
        self.text = new_text

    def getWordFreqs(self):
        return self.word_freqs

    def getTrueClass(self):
        return self.true_class

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess


# Read all text files in the given directory and construct the data set, D
# the directory path should just be like "train/ham" for example
# storage is the dictionary to store the email in
# True class is the true classification of the email (spam or ham)
def makeDataSet(storage_dict, directory, true_class):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                storage_dict.update({dir_entry_path: Document(text, bagOfWords(text), true_class)})


# counts frequency of each word in the text files and order of sequence doesn't matter
def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


# Extracts the vocabulary of all the text in a data set
def extractVocab(data_set):
    all_text = ""
    v = []
    for x in data_set:
        all_text += data_set[x].getText()
    
    for y in bagOfWords(all_text):
        v.append(y)
    return v


# Set the stop words
def setStopWords():
    stops = []
    with open('stop_words.txt', 'r') as txt:
        stops = (txt.read().splitlines())
    return stops

def removeStopWords(vocabulary):
    stops = []
    with open('stop_words.txt', 'r') as txt:
        stops = (txt.read().splitlines())

    new_vocabulary = []
    for t in vocabulary:
        if not t in stops:
            new_vocabulary.append(t)
    
    return new_vocabulary

def stemming(vocabulary):
    porter = PorterStemmer()

    new_vocabulary = []
    for t in vocabulary:
        stem_t = porter.stem(t)
        if not stem_t in new_vocabulary:
            new_vocabulary.append(stem_t)
    return new_vocabulary

def lemmatizing(vocabulary):
    lemmatizer = WordNetLemmatizer() 
    new_vocabulary = []
    for t in vocabulary:
        stem_t = lemmatizer.lemmatize(t)
        if not stem_t in new_vocabulary:
            new_vocabulary.append(stem_t)
    return new_vocabulary

# Training
def trainMultinomialNB(training, classes, priors, cond, options):
    v = extractVocab(training)

    if int(options['is_remove_stopword']):
        v = removeStopWords(v)

    # v is the vocabulary of the training set
    if options['processing_type'] == 'stemming':
        v = stemming(v)
    elif options['processing_type'] == 'lemmatizing':
        v = lemmatizing(v)
        
    # n is the number of documents
    n = len(training)
    # for each class in classes (i.e. ham and spam)
    for c in classes:
        # n_c is number of documents with true class c
        n_c = 0.0
        # text_c = concatenation of text of all docs in class (D, c)
        text_c = ""
        for i in training:
            if training[i].getTrueClass() == c:
                n_c += 1
                text_c += training[i].getText()
        priors[c] = float(n_c) / float(n)
        # Count frequencies/tokens of each term in text_c in dictionary form (i.e. token : frequency)
        token_freqs = bagOfWords(text_c)
        # Calculate conditional probabilities for each token and sum using laplace smoothing and log-scale
        for t in v:
            if t in token_freqs:
                cond.update({t + "_" + c: (float((token_freqs[t] + 1.0)) / float((len(text_c) + len(v))))})
            else:
                cond.update({t + "_" + c: (float(1.0) / float((len(text_c) + len(v))))})
        
    # store dict in json file
    with open('learned/multinomial_conditional_probability.json', 'w') as fp:
        json.dump(cond, fp)
        
    with open('learned/prior.json', 'w') as fp:
        json.dump(priors, fp)

# Testing. Data instance is a Document
# Returns classification guess
def applyMultinomialNB(data_instance, classes, priors, cond):
    score = {"spam": 0, "ham": 0}
    for c in classes:
        score[c] = float(math.log10(priors[c]))
        for t in data_instance.getWordFreqs():
            if (t + "_" + c) in cond:
                score[c] += float(math.log10(cond[t + "_" + c]))
    
    if score["spam"] > score["ham"]:
        return "spam"
    else:
        return "ham"
    # return score['ham']

# Training
def trainBernoulliNB(training, classes, priors, cond, options):
    nums = {}
    # v is the vocabulary of the training set
    v = extractVocab(training)

    if int(options['is_remove_stopword']):
        v = removeStopWords(v)

    # v is the vocabulary of the training set
    if options['processing_type'] == 'stemming':
        v = stemming(v)
    elif options['processing_type'] == 'lemmatizing':
        v = lemmatizing(v)

    # n is the number of documents
    n = len(training)
    # for each class in classes (i.e. ham and spam)

    for c in classes:
        # n_c is number of documents with true class c
        n_c = 0.0
        text_c = ''
        for i in training:
            if training[i].getTrueClass() == c:
                n_c += 1
                words = training[i].getText().split()
                text_c += " ".join(sorted(set(words), key=words.index))
                
        token_freqs = bagOfWords(text_c)

        for t in v:
            nums[t + '_' + c] = 0
            if t in token_freqs:
                nums[t + '_' + c] = token_freqs[t]
            
            cond.update({t + "_" + c: (float((nums[t + '_' + c] + 1.0)) / float((n_c + 2.0)))})

        priors[c] = float(n_c) / float(n)
            
    # store dict in json file
    with open('learned/vocabulary.json', 'w') as fp:
        json.dump(v, fp)

    with open('learned/bernoulli_conditional_probability.json', 'w') as fp:
        json.dump(cond, fp)
        
    with open('learned/prior.json', 'w') as fp:
        json.dump(priors, fp)

def applyBernoulliNB(data_instance, classes, priors, cond):
    score = {}
    with open('learned/vocabulary.json', 'r') as f:
        v = json.load(f)

    tokens = data_instance.getWordFreqs()
    for c in classes:
        score[c] = float(math.log10(priors[c]))

        for t in v:
            if t in tokens:
                score[c] += float(math.log10(cond[t + '_' + c]))
            else:
                score[c] += float(math.log10(1 - cond[t + '_' + c]))
    
    if(score['spam'] > score['ham']):
        return 'spam'
    else:
        return 'ham'

