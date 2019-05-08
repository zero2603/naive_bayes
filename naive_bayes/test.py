import os
import json
import helper

training_set = dict()

# print(os.listdir('./emails/train/ham'))

helper.makeDataSet(training_set, '../emails/train/ham', 'ham')
helper.makeDataSet(training_set, '../emails/train/spam', 'spam')

# with open('../learned/training_set.json', 'w') as fp:
#     json.dump(training_set, fp)

print(training_set)