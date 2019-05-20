import os
from random import shuffle
from math import floor

def train_test_split_from_dir(datadir):
    print('================> split')
    all_files = os.listdir(os.path.abspath(datadir))
    data_files = list(filter(lambda f: f.endswith('.txt'), all_files))
    shuffle(data_files)

    split = 0.8
    split_index = int(len(data_files) * split)
    training = data_files[:split_index]
    testing = data_files[split_index:]
    return training, testing