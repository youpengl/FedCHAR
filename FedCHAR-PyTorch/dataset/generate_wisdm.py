import numpy as np
import random
import os
import pandas as pd
from scipy import stats
from utils.HAR_utils import *
from utils.plot_utils import plot_distribution

random.seed(1)
np.random.seed(1)
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

CLASS_SET = [
    'Walking',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Downstairs'
]

DATA_PATH = 'dataset/wisdm/rawdata/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt'
dir_path = "dataset/wisdm/"
config_path = dir_path + "config.json"
train_path = dir_path + "train/"
test_path = dir_path + "test/"
SEGMENT_TIME_SIZE = 200
TIME_STEP = SEGMENT_TIME_SIZE

# LOAD DATA
data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, on_bad_lines='skip')
data['z-axis'].replace({';': ''}, regex=True, inplace=True)
data = data.dropna()
v_c = data['user'].value_counts()
# print(v_c.index.asi8, v_c.values)


X = []
Y = []

for idx, user_id in enumerate(sorted(v_c.index)):
    # DATA PREPROCESSING
    data_convoluted = []
    labels = []
    user_data = data[data['user'] == user_id]
    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
    for i in range(0, len(user_data) - SEGMENT_TIME_SIZE, TIME_STEP):
        x = user_data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
        y = user_data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
        z = user_data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
        data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
        label = stats.mode(user_data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
        labels.append(CLASS_SET.index(label))
    
    # Convert to numpy
    data_convoluted = np.asarray(data_convoluted, dtype=np.float32).reshape(-1, 3, 1, 200) # (N, 200 ,3) # (N, 9, 1, 128) 
    labels = np.array(labels)

    shuffle_idx = np.random.choice(np.arange(labels.shape[0]), labels.shape[0], replace=False)
    data_convoluted = data_convoluted[shuffle_idx]
    labels = labels[shuffle_idx]

    # print("Convoluted data shape: ", data_convoluted.shape)
    X.append(data_convoluted)
    Y.append(labels)


statistic = []
num_clients = len(Y)
num_classes = len(np.unique(np.concatenate(Y, axis=0)))
for i in range(num_clients):
    statistic.append([])
    for yy in sorted(np.unique(Y[i])):
        idx = Y[i] == yy
        statistic[-1].append((int(yy), int(len(X[i][idx]))))

for i in range(num_clients):
    print(f"Client {i}\t Size of data: {len(X[i])}\t Labels: ", np.unique(Y[i]))
    print(f"\t\t Samples of labels: ", [i for i in statistic[i]])
    print("-" * 50)


train_data, test_data = split_data(X, Y)

client_labels = [c_data['y'] for c_data in train_data]
plot_distribution(client_labels, num_classes, num_clients, CLASS_SET, 'WISDM')

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic)