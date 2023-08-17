import os
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
import math
import itertools
from .Newbase import BaseFedarated
from flearn.utils.tf_utils import get_stdev, l2_clip


class Server(BaseFedarated):#------------1
    def __init__(self, params, learner, dataset):
        Adam_Dataset = ['FMCW', 'WISDM', 'MobiAct']
        if params['dataset'] in Adam_Dataset:
            self.inner_opt = tf.train.AdamOptimizer(params['learning_rate'])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        self.num_classes = params['model_params'][0]
        super(Server, self).__init__(params, learner, dataset)
        self.params = params

    def train(self):
        clients_per_round = math.ceil(len(self.clients) * self.participation_ratio)
        print('---{} clients per communication round---'.format(clients_per_round))
        np.random.seed(self.corrupted_seed)
        num_of_corrupted = int(len(self.clients) * self.corrupted_ratio)
        corrupt_id = np.random.choice(range(len(self.clients)), size=num_of_corrupted, replace=False)
        print("Total {} corrupt_id: {}".format(num_of_corrupted, corrupt_id))

        for idx, c in enumerate(self.clients):
            if idx in corrupt_id and self.attack_type == 'A1':
                c.train_data['y'] = np.asarray(c.train_data['y'])
                if self.dataset == 'HARBox':
                    c.train_data['y'] = np.random.randint(0, 5, len(c.train_data['y']))
                elif self.dataset == 'IMU':
                    c.train_data['y'] = np.random.randint(0, 3, len(c.train_data['y']))
                elif self.dataset == 'Depth':
                    c.train_data['y'] = np.random.randint(0, 5, len(c.train_data['y']))
                elif self.dataset == 'UWB':
                    c.train_data['y'] = 1 - c.train_data['y']
                elif self.dataset == 'FMCW':
                    c.train_data['y'] = np.random.randint(0, 6, len(c.train_data['y']))
                elif self.dataset == 'WISDM':
                    c.train_data['y'] = np.random.randint(0, 6, len(c.train_data['y']))
                elif self.dataset == 'MobiAct':
                    c.train_data['y'] = np.random.randint(0, 7, len(c.train_data['y']))
        INFO = []
        #initial_rounds
        random_seed = 0
        for i in range(self.initial_rounds):

            indices, selected_clients = self.select_clients(round=i, num_clients=clients_per_round)
            print("initial_round {} selected clients indices: {}".format(i, indices))
            if self.attack_type != 'B':
                selected_corrupt_id = list(set(indices) & set(corrupt_id))
                selected_benign_id = np.setdiff1d(indices, selected_corrupt_id)
                print('selected_benign_id: {}'.format(list(selected_benign_id)))
                print('selected_corrupt_id: {}'.format(selected_corrupt_id))

            csolns = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
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
                        _, grads, _ = c.solve_sgd(data_batch)  #weights,grads,loss

                        if self.dynamic_lamda:

                            model_tmp = copy.deepcopy(self.local_models[idx])
                            model_best = copy.deepcopy(self.local_models[idx])
                            tmp_loss = 10000
                            # pick a lambda locally based on validation data
                            for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
                                for layer in range(len(grads[1])):
                                    eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.global_model[layer])
                                    model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                                c.set_params(model_tmp)
                                l = c.get_val_loss()
                                if l < tmp_loss:
                                    tmp_loss = l
                                    model_best = copy.deepcopy(model_tmp)

                            self.local_models[idx] = copy.deepcopy(model_best)

                        else:
                            for layer in range(len(grads[1])):
                                eff_grad = grads[1][layer] + self.lamda * (self.local_models[idx][layer] - self.global_model[layer])#vk= vk− η(∇Fk(vk) + λ(vk− wt))
                                self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                        # global
                        self.client_model.set_params(w_global_idx)
                        #loss = c.get_loss()
                        _, grads, _ = c.solve_sgd(data_batch)
                        w_global_idx = self.client_model.get_params()

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]


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
                csolns.append(diff)

            if self.Robust == 1:
                csolns = l2_clip(csolns)

            expected_num_mali = int(clients_per_round * self.corrupted_ratio)

            if self.Robust == 2:
                avg_updates = self.median_average(csolns)
            elif self.Robust == 3:
                avg_updates = self.k_norm_average(clients_per_round - expected_num_mali, csolns)
            elif self.Robust == 4:
                avg_updates = self.krum_average(clients_per_round - expected_num_mali - 2, csolns)
            elif self.Robust == 5:
                m = clients_per_round - expected_num_mali
                avg_updates = self.mkrum_average(clients_per_round - expected_num_mali - 2, m, csolns)
            else:
                avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]

            info, _ = self.evaluation(corrupt_id)
            INFO.append(info)
        print("initial rounds are over")
        csolns = []

        #clustering stage
        user_belong_g = np.ones([len(self.clients)]).astype('int') * -1
        # if self.num_of_clusters != 1:
        state = np.zeros([len(self.clients)]).astype(int)
        indices, selected_clients = self.select_clients(round=self.initial_rounds, num_clients=clients_per_round)
        print("clustering stage selected clients indices: {}".format(indices))
        if self.attack_type != 'B':
            selected_corrupt_id = list(set(indices) & set(corrupt_id))
            selected_benign_id = np.setdiff1d(indices, selected_corrupt_id)
            print('selected_benign_id: {}'.format(list(selected_benign_id)))
            print('selected_corrupt_id: {}'.format(selected_corrupt_id))

        for idx in indices:
            w_global_idx = copy.deepcopy(self.global_model)
            c = self.clients[idx]
            for epoch_i in range(self.epoch):
                num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
                random_seed = random_seed + 1
                for s in range(num_batch):
                    batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                    batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                    data_batch = (batch_xs, batch_ys)
                    # global
                    self.client_model.set_params(w_global_idx)
                    #loss = c.get_loss()
                    _, grads, _ = c.solve_sgd(data_batch)
                    w_global_idx = self.client_model.get_params()
                    # local
                    self.client_model.set_params(self.local_models[idx])
                    _, grads, _ = c.solve_sgd(data_batch)
                    if self.dynamic_lamda:

                        model_tmp = copy.deepcopy(self.local_models[idx])
                        model_best = copy.deepcopy(self.local_models[idx])
                        tmp_loss = 10000
                        # pick a lambda locally based on validation data
                        for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
                            for layer in range(len(grads[1])):
                                eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.global_model[layer])
                                model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            c.set_params(model_tmp)
                            l = c.get_val_loss()
                            if l < tmp_loss:
                                tmp_loss = l
                                model_best = copy.deepcopy(model_tmp)

                        self.local_models[idx] = copy.deepcopy(model_best)
                    else:
                        for layer in range(len(grads[1])):
                            eff_grad = grads[1][layer] + self.lamda * (self.local_models[idx][layer] - self.global_model[layer])#vk= vk− η(∇Fk(vk) + λ(vk− wt))
                            self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

            # get the difference (global model updates)
            diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

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
            csolns.append(diff)
        info, _ = self.evaluation(corrupt_id)
        INFO.append(info)
        # server performs cluster and then update group models
        G, Z = self.hierarchical_cluster(csolns, indices)
        # Save csolns per cluster
        G_csolns = []

        for g_idx, g in enumerate(G):
            g_csolns = [csolns[list(indices).index(gg)] for gg in g]
            G_csolns.append(g_csolns)
            user_belong_g[g] = g_idx

        AvgG_csolns = []
        G_models = [self.global_model for g in range(len(G))]
        for g in range(len(G)):
            avg_updates = self.simple_average(G_csolns[g])
            AvgG_csolns.append(avg_updates)
            for layer in range(len(avg_updates)):
                G_models[g][layer] += avg_updates[layer]
        # else:
        #     G = [[i for i in range(len(self.clients))]]
        #     Z = []
        print("Groups info: {}".format(G))
        print("\n")


        for i in range(self.remain_rounds - 1):
            indices, selected_clients = self.select_clients(round=self.initial_rounds+1+i, num_clients=clients_per_round)
            print("remain round {} selected clients indices: {}".format(i, indices))
            non_indices = np.setdiff1d(range(len(self.clients)), indices)
            if self.attack_type != 'B':
                selected_corrupt_id = list(set(indices) & set(corrupt_id))
                selected_benign_id = np.setdiff1d(indices, selected_corrupt_id)
                print('selected_benign_id: {}'.format(list(selected_benign_id)))
                print('selected_corrupt_id: {}'.format(selected_corrupt_id))
            state[non_indices] += 1
            csolns = []
            for idx in indices:
                if user_belong_g[idx] != -1 and state[idx] < self.recluster_rounds:
                    g_model = copy.deepcopy(G_models[user_belong_g[idx]])
                    diff, random_seed = self.local_train(g_model, idx, random_seed, corrupt_id)
                    csolns.append(diff)
                else:
                    w_global_idx = copy.deepcopy(self.global_model)
                    for epoch_i in range(self.epoch):
                        num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                        train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
                        random_seed = random_seed + 1
                        for s in range(num_batch):
                            batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                            batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                            data_batch = (batch_xs, batch_ys)
                            # global
                            self.client_model.set_params(w_global_idx)
                            #loss = c.get_loss()
                            _, grads, _ = c.solve_sgd(data_batch)
                            w_global_idx = self.client_model.get_params()
                    diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]
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
                    g_idx = self.caculate_similarity(diff, AvgG_csolns)
                    if g_idx == len(G_models):
                        g_model = copy.deepcopy(self.global_model)
                        G_models.append(g_model)
                        AvgG_csolns.append(diff)
                        diff, random_seed = self.local_train(g_model, idx, random_seed, corrupt_id)
                        csolns.append(diff)
                    else:
                        g_model = copy.deepcopy(G_models[g_idx])
                        diff, random_seed = self.local_train(g_model, idx, random_seed, corrupt_id)
                        csolns.append(diff)
                    user_belong_g[idx] = g_idx

            state[indices] = 0
            G_csolns = []
            G = []
            for g_idx in range(len(G_models)):
                g = []
                g_csolns = []
                g_where = np.where(user_belong_g == g_idx)[0]
                for id, idx in enumerate(indices):
                    if idx in g_where:
                        g_csolns.append(csolns[id])
                        g.append(idx)
                G_csolns.append(g_csolns)
                G.append(g)
            print('Remain round {} G info: {}'.format(i, G))
            for g_idx in range(len(G_csolns)):
                if G_csolns[g_idx] == []:
                    continue
                avg_updates = self.simple_average(G_csolns[g_idx])
                for layer in range(len(avg_updates)):
                    G_models[g_idx][layer] += avg_updates[layer]

            info, last_round_test_acc_per_user = self.evaluation(corrupt_id)
            INFO.append(info)

        G_end = []
        G_vol = 0
        for g_idx in range(max(user_belong_g)+1):
            group = np.where(user_belong_g == g_idx)[0].tolist()
            G_vol += len(group)
            G_end.append(group)
        print('Last length {} G: {}'.format(G_vol, G_end))
        print('Malicious nodes id: {}'.format(corrupt_id))
        INFO = np.array(INFO)
        train_acc_per_round = INFO[:, 0]
        test_acc_per_round = INFO[:, 1]
        train_loss_per_round = INFO[:, 2]
        test_loss_per_round = INFO[:, 3]
        variance_per_round = INFO[:, 4]
        benign_id = np.setdiff1d(range(len(self.clients)), corrupt_id)

        if not os.path.exists('./record/Dataset/{}'.format(self.dataset)):
            os.makedirs('./record/Dataset/{}'.format(self.dataset))

        np.savez('./record/Dataset/{}/FedCHAR_DC_ir{}_rr{}_c{}_pr{}_atk{}_cr{}_lr{}_bz{}_ep{}_rc{}.npz'.format(self.dataset,
                                                                                         self.initial_rounds,
                                                                                         self.remain_rounds,
                                                                                         self.num_of_clusters,
                                                                                         self.participation_ratio,
                                                                                         self.attack_type,
                                                                                         self.corrupted_ratio,
                                                                                         self.learning_rate,
                                                                                         self.batch_size,
                                                                                         self.epoch,
                                                                                         self.recluster_rounds),
                 train_acc_per_round=train_acc_per_round,
                 test_acc_per_round=test_acc_per_round,
                 train_loss_per_round=train_loss_per_round,
                 test_loss_per_round=test_loss_per_round,
                 variance_per_round=variance_per_round,
                 last_round_test_acc_per_user=last_round_test_acc_per_user,
                 benign_id=benign_id,
                 param=self.params)