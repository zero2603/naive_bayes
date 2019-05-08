from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.template import loader
import helper 
import json

# VARIABLE
classes = ["ham", "spam"]


# CONTROLLER
# return index page
def index(request):
    template = loader.get_template('training.html')
    context = {}
    return HttpResponse(template.render(context, request))

# def training(request):
#     template = loader.get_template('testing.html')
    
#     data = request.POST
#     context = helper.trainAndTest(int(data['is_stemming']), int(data['is_remove_stopword']), data['algorithm'], float(data['ratio']))

#     return HttpResponse(template.render(context, request))

# def test(request):
#     template = loader.get_template('result.html')
    
#     # data = request.POST
#     predictClass =  helper.testing()

#     return HttpResponse(template.render({'predictClass': predictClass}, request))

def training(request):
    training_set = dict()
    prior = dict()
    conditional_probability = dict()

    template = loader.get_template('testing.html')

    data = request.POST

    helper.makeDataSet(training_set, 'emails/train/ham', 'ham')
    helper.makeDataSet(training_set, 'emails/train/spam', 'spam')

    if(int(data['is_training'])):
        if data['algorithm'] == 'multinomial':
            helper.trainMultinomialNB(training_set, classes, prior, conditional_probability, data)
        else:
            helper.trainBernoulliNB(training_set, classes, prior, conditional_probability, data)

    context = {'ok': 1}
    return HttpResponse(template.render(context, request))

def test(request):
    template = loader.get_template('result.html')

    conditional_probability = dict()
    prior = dict()

    if current_algorithm == 'multinomial':
        with open('learned/multinomial_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
    else:
        with open('learned/bernoulli_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)
    
    with open('learned/prior.json', 'r') as f:
        prior = json.load(f)
    
    document = helper.Document(request.POST['testing_text'], helper.bagOfWords(request.POST['testing_text']), None)
    result = helper.applyMultinomialNB(document, classes, prior, conditional_probability)

    return HttpResponse(template.render({'result': result}, request))

def testMany(request):
    template = loader.get_template('result.html')

    test_set = dict()
    conditional_probability = dict()
    prior = dict()

    with open('learned/multinomial_conditional_probability.json', 'r') as f:
            conditional_probability = json.load(f)

    # if current_algorithm == 'multinomial':
    #     with open('learned/multinomial_conditional_probability.json', 'r') as f:
    #         conditional_probability = json.load(f)
    # else:
    #     with open('learned/bernoulli_conditional_probability.json', 'r') as f:
    #         conditional_probability = json.load(f)
    
    with open('learned/prior.json', 'r') as f:
        prior = json.load(f)

    helper.makeDataSet(test_set, 'emails/test/ham', 'ham')
    helper.makeDataSet(test_set, 'emails/test/spam', 'spam')



    correct_ham_guesses = 0
    correct_spam_guesses = 0

    for i in test_set:
        test_set[i].setLearnedClass(helper.applyMultinomialNB(test_set[i], classes, prior, conditional_probability))
        if test_set[i].getLearnedClass() == test_set[i].getTrueClass():
            if test_set[i].getTrueClass() == 'ham':
                correct_ham_guesses += 1
            if test_set[i].getTrueClass() == 'spam':
                correct_spam_guesses += 1

    return HttpResponse(template.render({'correct_ham_guesses': correct_ham_guesses, 'correct_spam_guesses': correct_spam_guesses}, request))