import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
from flearn.models.client import Client
from flearn.utils.tf_utils import process_grad
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class BaseFedarated(object):
    def __init__(self, params, learner, dataset):
        # transfer parameters to self
        for key, val in params.items():
            setattr(self, key, val)

        # create worker nodes
        tf.reset_default_graph()
        self.client_model = learner(*params['model_params'], self.q, self.inner_opt, self.seed)
        self.clients = self.setup_clients(dataset, self.dynamic_lamda, self.client_model)

        print('{} Clients in Total'.format(len(self.clients)))
        self.latest_model = copy.deepcopy(self.client_model.get_params())
        self.local_models = []
        self.global_model = copy.deepcopy(self.latest_model)
        for _ in self.clients:
            self.local_models.append(copy.deepcopy(self.latest_model))


        # # initialize system metrics
        # self.metrics = Metrics(self.clients, params)

    def __del__(self):
        self.client_model.close()

    def setup_clients(self, dataset, dynamic_lam, model=None):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]
        all_clients = [Client(u, g, train_data[u], test_data[u], dynamic_lam, model) for u, g in zip(users, groups)]
        return all_clients
    
    def train_error(self, models, group_c):
        num_samples = []
        tot_correct = []
        losses = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns = c.train_error()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)

        return np.array(num_samples), np.array(tot_correct), np.array(losses)

    def test(self, models, group_c):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        Group_cm = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            Group_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(Group_cm)

    def val(self, models, group_c):
        '''validate self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        Group_cm = []

        for idx, c in enumerate(group_c):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.val()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            Group_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(Group_cm)

    def save(self):
        pass

    def shuffle_training_data(self, train_data, train_labels, epoch_i):
        seed = epoch_i
        np.random.seed(seed)
        np.random.shuffle(train_data)
        np.random.seed(seed)
        np.random.shuffle(train_labels)
        return np.array(train_data), np.array(train_labels)

    def select_clients(self, round, num_clients=20):

        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round)

        if self.sampling == 1:
            pk = np.ones(num_clients) / num_clients
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices] 
        elif self.sampling == 2:
            num_samples = []
            for client in self.clients:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(range(len(self.clients)), num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

    def select_clients_intraG(self, round, num_clients, group, group_c):

        num_clients = min(num_clients, len(self.clients))

        np.random.seed(round)

        if self.sampling == 1:
            pk = np.ones(len(group)) / len(group)
            indices = np.random.choice(group, num_clients, replace=False, p=pk) 
            return indices, np.asarray(self.clients)[indices]

        elif self.sampling == 2:
            num_samples = []
            for client in group_c:
                num_samples.append(client.train_samples)
            total_samples = np.sum(np.asarray(num_samples))
            pk = [item * 1.0 / total_samples for item in num_samples]
            indices = np.random.choice(group, num_clients, replace=False, p=pk)
            return indices, np.asarray(self.clients)[indices]

    def hierarchical_cluster(self, csolns, indices):
        #X.shape -> (num_clients, dim_of_flatten(model))
        X = np.array([])
        for k in range(len(csolns[0])):
            X = np.concatenate((X, csolns[0][k].flatten()), axis=0)
        X = X.reshape(1, -1)
        for i in range(1, len(csolns)):
            v = np.array([])
            for k in range(len(csolns[0])):
                v = np.concatenate((v, csolns[i][k].flatten()), axis=0)
            v= v.reshape(1, -1)
            X = np.r_[X, v] #this reshape is invalid

        Z = linkage(X, method=self.linkage, metric=self.distance)
        print("Clustering Structure: {}".format(Z))
        # c, coph_dists = cophenet(Z, pdist(X))
        # print("cophenet: {}".format(c))
        # plt.figure(figsize=(10, 10))
        # # index = range(1, Z.shape[0]+1)
        # # plt.plot(index, Z[::-1][:, 2])
        # # plt.xticks(index, index)
        # # plt.show()
        # dendrogram(Z, truncate_mode='lastp', p=120, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=10, show_contracted=True)
        # plt.show()
        # max_d = self.max_d
        # labels = fcluster(Z, t=max_d, criterion='distance')
        k = self.num_of_clusters
        labels = fcluster(Z, t=k, criterion='maxclust')

        G = []
        for i in range(1, k+1):
            tmp = []
            for j in range(len(csolns)):
                if labels[j] == i:
                    tmp.append(indices[j])
            G.append(tmp)

        for i, g in enumerate(G):
            if g == []:
                G.pop(i)
        return G, Z

    def initial_aggregate(self, G, csolns, indices):
        for i in range(len(G)):  
            Layer = [0] * len(csolns[0])  

            for j in G[i]:  
                index = np.where(indices == j)[0][0]
                for k, v in enumerate(csolns[index]):
                    Layer[k] += v.astype(np.float64) * (1 / len(G[i]))
            tmp_model = copy.deepcopy(self.global_model)
            for layer in range(len(csolns[0])):
                tmp_model[layer] += Layer[layer]
            self.group_models.append(tmp_model)

    def intra_aggregate(self, csolns, client_group_id):

        for i in range(len(self.group_models)): 
            Layer = [0] * len(csolns[0])  
            index = np.where(client_group_id == i)[0]
            for j in index:
                for k, v in enumerate(csolns[j]):
                    Layer[k] += v.astype(np.float64) * (1 / index.shape[0])
            for layer, value in enumerate(Layer):
                self.group_models[i][layer] = self.group_models[i][layer] + value


    def aggregate(self, wsolns):

        total_weight = 0.0
        base = [0] * len(wsolns[0][1])

        for (w, soln) in wsolns:
            total_weight += w
            for i, v in enumerate(soln):
                base[i] += w * v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def simple_average(self, parameters):

        base = [0] * len(parameters[0])

        for p in parameters:  # for each client
            for i, v in enumerate(p):
                base[i] += v.astype(np.float64)   # the i-th layer

        averaged_params = [v / len(parameters) for v in base]

        return averaged_params

    def median_average(self, parameters):

        num_layers = len(parameters[0])
        aggregated_models = []
        for i in range(num_layers):
            a = []
            for j in range(len(parameters)):
                a.append(parameters[j][i].flatten())
            aggregated_models.append(np.reshape(np.median(a, axis=0), newshape=parameters[0][i].shape))

        return aggregated_models

    def krum_average(self, k, parameters):
        # krum: return the parameter which has the lowest score defined as the sum of distance to its closest k vectors
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i+1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k+1])  # the distance including itself, so k+1 not k

        selected_idx = np.argsort(score)[0]

        return parameters[selected_idx]

    def mkrum_average(self, k, m, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        distance = np.zeros((len(parameters), len(parameters)))
        for i in range(len(parameters)):
            for j in range(i + 1, len(parameters)):
                distance[i][j] = np.sum(np.square(flattened_grads[i] - flattened_grads[j]))
                distance[j][i] = distance[i][j]

        score = np.zeros(len(parameters))
        for i in range(len(parameters)):
            score[i] = np.sum(np.sort(distance[i])[:k + 1])  # the distance including itself, so k+1 not k

        # multi-krum selects top-m 'good' vectors (defined by socre) (m=1: reduce to krum)
        selected_idx = np.argsort(score)[:m]
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])

        return self.simple_average(selected_parameters)


    def k_norm_average(self, num_benign, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        norms = [np.linalg.norm(u) for u in flattened_grads]
        selected_idx = np.argsort(norms)[:num_benign]  # filter out the updates with large norms
        selected_parameters = []
        for i in selected_idx:
            selected_parameters.append(parameters[i])
        return self.simple_average(selected_parameters)

    def k_loss_average(self, num_benign, losses, parameters):
        flattened_grads = []
        for i in range(len(parameters)):
            flattened_grads.append(process_grad(parameters[i]))
        selected_idx = np.argsort(losses)[num_benign-1]  # select the update with largest loss among the benign devices
        return parameters[selected_idx]