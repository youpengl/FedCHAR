import json
import numpy as np
import os

def gen_batch(data, batch_size, num_iter):

    data_x = np.array(data['x'])
    data_y = np.array(data['y'])

    index = len(data_y)

    for i in range(num_iter):
        index += batch_size
        if (index + batch_size > len(data_y)):
            index = 0
            np.random.seed(i+1)
            # randomly shuffle the data after one pass of the entire training set         
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index + batch_size]
        batched_y = data_y[index: index + batch_size]
        
        yield (batched_x, batched_y)

def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    clients = []
    groups = []#del
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])#["f_00010", "f_00011", "f_00012", ... ,]
        if 'hierarchies' in cdata:#del
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])#{"f_00010":{'y':[],'x':[[]]}...}


    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data
