import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
from flearn.models.client import Client
from flearn.utils.tf_utils import process_grad
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet
from sklearn.metrics.pairwise import cosine_similarity
import math
from flearn.utils.tf_utils import get_stdev
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

    def local_train(self, g_model, idx, random_seed, corrupt_id):
        w_global_idx = copy.deepcopy(g_model)
        c = self.clients[idx]
        for epoch_i in range(self.epoch):
             num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
             train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
             random_seed = random_seed + 1
             for s in range(num_batch):
                 batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                 batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                 data_batch = (batch_xs, batch_ys)
                 # local
                 self.client_model.set_params(self.local_models[idx])
                 _, grads, _ = c.solve_sgd(data_batch)  # weights,grads,loss

                 for layer in range(len(grads[1])):
                     eff_grad = grads[1][layer] + self.lamda * (self.local_models[idx][layer] - g_model[layer])
                     self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                 # global
                 self.client_model.set_params(w_global_idx)
                 _, grads, _ = c.solve_sgd(data_batch)
                 w_global_idx = self.client_model.get_params()
        # get the difference (global model updates)
        diff = [u - v for (u, v) in zip(w_global_idx, g_model)]
         # send the malicious updates
        if idx in corrupt_id:
            if self.attack_type == 'A3':
                # scale malicious updates
                diff = [10 * u for u in diff]
            elif self.attack_type == 'A2':
                # send random updates
                stdev_ = get_stdev(diff)
                diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]
            elif self.attack_type == 'A4':
                diff = [-1 * u for u in diff]
        return diff, random_seed
    def evaluation(self, corrupt_id):
        benign_id = np.setdiff1d(range(len(self.clients)), corrupt_id)
        tmp_models = []
        for idx in range(len(self.clients)):
            tmp_models.append(self.local_models[idx])
        num_train, num_correct_train, train_loss_vector = self.train_error(tmp_models)
        num_test, num_correct_test, test_loss_vector, ALL_CM = self.test(tmp_models)
        # if val
        # num_val, num_correct_val, val_loss_vector, ALL_CM = self.val(tmp_models)
        avg_train_acc = np.sum(num_correct_train[benign_id]) * 1.0 / np.sum(num_train[benign_id])
        avg_train_loss = np.dot(train_loss_vector[benign_id], num_train[benign_id]) / np.sum(num_train[benign_id])
        avg_test_acc = np.sum(num_correct_test[benign_id]) * 1.0 / np.sum(num_test[benign_id])
        avg_test_loss = np.dot(test_loss_vector[benign_id], num_test[benign_id]) / np.sum(num_test[benign_id])
        train_acc_per_user = num_correct_train[benign_id] / num_train[benign_id]
        test_acc_per_user = num_correct_test[benign_id] / num_test[benign_id]
        variance = np.var(test_acc_per_user)
        print('benign_avg_train_acc: {}'.format(avg_train_acc))
        print('benign_avg_test_acc: {}'.format(avg_test_acc))
        print('benign_train_acc_per_user: {}'.format(train_acc_per_user.tolist()))
        print('benign_test_acc_per_user: {}'.format(test_acc_per_user.tolist()))
        print('benign_variance: {}'.format(variance))
        print('\n')
        info = []
        info.append(avg_train_acc)
        info.append(avg_test_acc)
        info.append(avg_train_loss)
        info.append(avg_test_loss)
        info.append(variance)
        return info, test_acc_per_user.tolist()
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
    def test(self, models):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        ALL_cm = []
        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.test()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            ALL_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(ALL_cm)
    def val(self, models):
        '''val self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        losses = []
        ALL_cm = []
        for idx, c in enumerate(self.clients):
            self.client_model.set_params(models[idx])
            ct, cl, ns, cm = c.val()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)
            ALL_cm.append(cm)
        return np.array(num_samples), np.array(tot_correct), np.array(losses), np.array(ALL_cm)
    def train_error(self, models):
            num_samples = []
            tot_correct = []
            losses = []

            for idx, c in enumerate(self.clients):
                self.client_model.set_params(models[idx])
                ct, cl, ns = c.train_error()
                tot_correct.append(ct * 1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)

            return np.array(num_samples), np.array(tot_correct), np.array(losses)
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
        # plt.figure(figsize=(10, 10))
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
    def caculate_similarity_new(self, diff, AvgG_csolns): #avg/median
        max = -1
        min = 999
        g_belong = -1
        main = np.array([])
        for k in range(len(diff)):
            main = np.concatenate((main, diff[k].flatten()), axis=0)
        for g_idx, csolns_i in enumerate(AvgG_csolns):
            for g_jdx, csolns_j in enumerate(AvgG_csolns):
                if g_idx < g_jdx:
                    v1 = np.array([])
                    v2 = np.array([])
                    for k in range(len(csolns_i)):
                        v1 = np.concatenate((v1, csolns_i[k].flatten()), axis=0)
                        v2 = np.concatenate((v2, csolns_j[k].flatten()), axis=0)
                    # v1 = v1.reshape(-1, 1)
                    # v2 = v2.reshape(-1, 1)
                    cos_sim = v1.dot(v2) / np.linalg.norm(v1) * np.linalg.norm(v2)
                    if cos_sim < min:
                        min = cos_sim

        for g_idx, csolns in enumerate(AvgG_csolns):
            other = np.array([])
            for k in range(len(csolns)):
                other = np.concatenate((other, csolns[k].flatten()), axis=0)
            cos_sim = main.dot(other) / np.linalg.norm(main) * np.linalg.norm(other)
            if cos_sim > max:
                max = cos_sim
                g_belong = g_idx
        if max < min:
            print('Min cosine similarity within G: {}'.format(min))
            print('Max cosine similarity new user to G: {}'.format(max))
            return len(AvgG_csolns)
        else:
            return g_belong
    def caculate_similarity(self, diff, AvgG_csolns): #avg/median
        max = -1
        min = 999
        g_belong = -1
        main = np.array([])
        for k in range(len(diff)):
            main = np.concatenate((main, diff[k].flatten()), axis=0)
        main = main.reshape(1, -1)
        for g_idx, csolns_i in enumerate(AvgG_csolns):
            for g_jdx, csolns_j in enumerate(AvgG_csolns):
                if g_idx < g_jdx:
                    v1 = np.array([])
                    v2 = np.array([])
                    for k in range(len(csolns_i)):
                        v1 = np.concatenate((v1, csolns_i[k].flatten()), axis=0)
                        v2 = np.concatenate((v2, csolns_j[k].flatten()), axis=0)
                    v1 = v1.reshape(1, -1)
                    v2 = v2.reshape(1, -1)
                    cos_sim = cosine_similarity(v1, v2)[0][0]

                    if cos_sim < min:
                        min = cos_sim

        for g_idx, csolns in enumerate(AvgG_csolns):
            other = np.array([])
            for k in range(len(csolns)):
                other = np.concatenate((other, csolns[k].flatten()), axis=0)
            other = other.reshape(1, -1)
            cos_sim = cosine_similarity(main, other)[0][0]
            if cos_sim > max:
                max = cos_sim
                g_belong = g_idx
        if len(AvgG_csolns) == 1:
            if max > 0.:
                return g_belong
            else:
                print('Cosine similarity new user to only one cluster: {}'.format(max))
                return len(AvgG_csolns)

        else:
            if max < min:
                print('Min cosine similarity within G: {}'.format(min))
                print('Max cosine similarity new user to G: {}'.format(max))
                return len(AvgG_csolns)
            else:
                return g_belong

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
