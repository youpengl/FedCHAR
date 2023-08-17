import matplotlib.pyplot as plt
import numpy as np

# edgecolor = ['#6495ed', '#8fbc8b', '#cd5c5c', '#ffa07a', '#9370db', '#b22222', '#ffd700']
edgecolor = ['#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000', '#000000'] * 10
color = ['#A6CEE3', '#B2DF8A', '#FB9A99', '#FDBF6F', '#CAB2D6', '#FF7F00', '#FFF68F', '#1f77b4', '#ff7f0e', '#2ca02c'] * 10
hatch = ['---', '///', '\\\\\\', 'xxx', '|||', '...', '<<<', '~~~', '===', None] * 10

def plot_distribution(client_labels, num_classes, num_clients, class_set, dataset):
    num_samples = np.array([label.shape[0] for label in client_labels])
    num_samples_index = np.argsort(-num_samples)
    arr = np.zeros([num_clients, num_classes])
    for idx, user_id in enumerate(num_samples_index):
        for class_id in range(num_classes):
            arr[idx][class_id] = client_labels[user_id].tolist().count(class_id)

    arr = arr.T
    x = ['Client {}'.format(i) for i in num_samples_index]
    fig = plt.figure(figsize=(10, 5))

    temp = np.zeros(shape=arr[0].shape)
    for class_idx in range(num_classes):
        if class_idx == 0:
            plt.bar(x, arr[class_idx], color=color[class_idx], hatch=hatch[class_idx], alpha=0.99, edgecolor=edgecolor[class_idx], label=class_set[class_idx])

        else:
            plt.bar(x, arr[class_idx], bottom=temp, color=color[class_idx], hatch=hatch[class_idx], alpha=0.99, edgecolor=edgecolor[class_idx], label=class_set[class_idx])
        temp += arr[class_idx]
    
    if num_classes <= 10:
        plt.legend(loc='upper right', fontsize=18)
    plt.gca().axes.xaxis.set_ticks([])
    plt.xlabel('User ID', fontsize=20)
    plt.ylabel('Num of Samples', fontsize=20)
    plt.title('{}'.format(dataset), fontsize=20)
    plt.tick_params(labelsize=20)
    fig.tight_layout()
    # plt.show()
    plt.savefig('dataset/{}/{}.pdf'.format(str.lower(dataset), dataset), bbox_inches='tight', pad_inches=0.0)