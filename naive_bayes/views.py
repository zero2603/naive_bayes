from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.template import loader
import helper 
import json
import random
import processing

# VARIABLE
classes = ["ham", "spam"]

# CONTROLLER
# return index page
def index(request):
    # TRAIN AND TEST SET
    ham_training_set, ham_testing_set = processing.train_test_split_from_dir('emails/all/ham')
    spam_training_set, spam_testing_set = processing.train_test_split_from_dir('emails/all/spam')

    request.session['ham_training'] = ham_training_set
    request.session['spam_training'] = spam_training_set
    request.session['ham_testing'] = ham_testing_set
    request.session['spam_testing'] = spam_testing_set

    template = loader.get_template('training.html')
    context = {}
    return HttpResponse(template.render(context, request))

def training(request):

    training_set = dict()
    prior = dict()
    conditional_probability = dict()

    template = loader.get_template('testing.html')

    data = request.POST
    

    if(int(data['is_training'])):
        helper.makeDataSet(training_set, 'emails/all/ham', request.session.get('ham_training'), 'ham')
        helper.makeDataSet(training_set, 'emails/all/spam', request.session.get('spam_training'), 'spam')

        if data['algorithm'] == 'multinomial':
            helper.trainMultinomialNB(training_set, classes, prior, conditional_probability, data)
        else:
            helper.trainBernoulliNB(training_set, classes, prior, conditional_probability, data)

    context = {
        'is_remove_stopword': int(data['is_remove_stopword']),
        'processing_type': data['processing_type'],
        'algorithm': data['algorithm']
    }
    return HttpResponse(template.render(context, request))

def test(request):
    template = loader.get_template('result.html')

    conditional_probability = dict()
    prior = dict()

    data = request.POST
    
    with open('learned/prior.json', 'r') as f:
        prior = json.load(f)

    words = []
    new_text = ''
    temp = helper.bagOfWords(request.POST['testing_text'])
    for t in temp:
        words.append(t)

    if int(data['is_remove_stopword']):
        words = helper.removeStopWords(words)
    
    if data['processing_type'] == 'stemming':
        words = helper.stemming(words)
    elif data['processing_type'] == 'lemmatizing':
        words = helper.lemmatizing(words)

    new_text += " ".join(words)
    
    document = helper.Document(new_text, helper.bagOfWords(new_text), None)

    if data['algorithm'] == 'multinomial':
        with open('learned/multinomial_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
        result = helper.applyMultinomialNB(document, classes, prior, conditional_probability)
    else:
        with open('learned/bernoulli_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
        result = helper.applyBernoulliNB(document, classes, prior, conditional_probability)

    return HttpResponse(template.render({'result': result}, request))

def testMany(request):
    data = request.POST
    template = loader.get_template('result.html')

    test_set = dict()
    conditional_probability = dict()
    prior = dict()

    if data['algorithm'] == 'multinomial':
        with open('learned/multinomial_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
    else:
        with open('learned/bernoulli_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
    
    with open('learned/prior.json', 'r') as f:
        prior = json.load(f)

    helper.makeDataSet(test_set, 'emails/all/ham', request.session.get('ham_testing'), 'ham')
    helper.makeDataSet(test_set, 'emails/all/spam', request.session.get('spam_testing'), 'spam')

    correct_ham_guesses = 0
    correct_spam_guesses = 0

    for i in test_set:
        words = []
        new_text = ''
        temp = helper.bagOfWords(test_set[i].getText())
        for t in temp:
            words.append(t)

        if int(data['is_remove_stopword']):
            words = helper.removeStopWords(words)

        if data['processing_type'] == 'stemming':
            words = helper.stemming(words)
        elif data['processing_type'] == 'lemmatizing':
            words = helper.lemmatizing(words)
        
        new_text += " ".join(words)
        test_set[i].setText(new_text)

        if data['algorithm'] == 'multinomial':
            test_set[i].setLearnedClass(helper.applyMultinomialNB(test_set[i], classes, prior, conditional_probability))
        else:
            test_set[i].setLearnedClass(helper.applyBernoulliNB(test_set[i], classes, prior, conditional_probability))

        if test_set[i].getLearnedClass() == test_set[i].getTrueClass():
            if test_set[i].getTrueClass() == 'ham':
                correct_ham_guesses += 1
            if test_set[i].getTrueClass() == 'spam':
                correct_spam_guesses += 1
        
        context = {
            'correct_ham_guesses': correct_ham_guesses,
            'ham_len': len(request.session.get('ham_testing')),
            'correct_spam_guesses': correct_spam_guesses,
            'spam_len': len(request.session.get('spam_testing')),
            'accuracy': float(correct_ham_guesses + correct_spam_guesses) / float(len(test_set))
        }

    return HttpResponse(template.render(context, request))