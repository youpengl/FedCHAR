import os
import numpy as np
from tqdm import tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import copy
import math
from .CHARbase import BaseFedarated
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
        attack_belong_dict = {}
        if self.attack_type == 'Hybrid':
            attack_belong = np.random.randint(1, 4+1, num_of_corrupted, dtype=int)
            for atk in range(1, 4+1):
                attack_belong_dict['A{}'.format(atk)] = corrupt_id[np.where(attack_belong == atk)[0].tolist()]
            print(attack_belong_dict)
        for idx, c in enumerate(self.clients):
            if idx in corrupt_id and (self.attack_type == 'A1' or (
                    self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 1)):
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

        random_seed = 0

        benign_test_acc = []
        benign_train_acc = []
        var_of_performance = []
        benign_test_loss = []
        benign_train_loss = []
        #initial_train
        for i in range(self.initial_rounds + 1):
            if i % self.eval_every == 0 and i > 0:
                tmp_models = []
                for idx in range(len(self.clients)):
                    tmp_models.append(self.local_models[idx])

                num_train, num_correct_train, train_loss_vector = self.train_error(tmp_models, self.clients)
                num_test, num_correct_test, test_loss_vector, Group_cm = self.test(tmp_models, self.clients)
                # if val
                # num_val, num_correct_val, val_loss_vector, Group_cm = self.val(tmp_models, self.clients)
                train_acc_once = np.sum(num_correct_train) * 1.0 / np.sum(num_train)
                avg_train_loss = np.dot(train_loss_vector, num_train) / np.sum(num_train)

                test_acc_once = np.sum(num_correct_test) * 1.0 / np.sum(num_test)
                avg_test_loss = np.dot(test_loss_vector, num_test) / np.sum(num_test)

                tqdm.write(
                    'At round {} training accu: {}, loss: {}'.format(i, train_acc_once, avg_train_loss))
                tqdm.write('At round {} test accu: {}, l1oss: {}'.format(i, test_acc_once, avg_test_loss))

                c_id = copy.deepcopy(corrupt_id)
                nc_id = np.setdiff1d(range(len(self.clients)), corrupt_id)

                malicious_train_loss_once = np.dot(train_loss_vector[c_id], num_train[c_id]) / np.sum(
                    num_train[c_id])
                benign_train_loss_once = np.dot(train_loss_vector[nc_id], num_train[nc_id]) / np.sum(
                    num_train[nc_id])
                malicious_test_loss_once = np.dot(test_loss_vector[c_id], num_test[c_id]) / np.sum(
                    num_test[c_id])
                benign_test_loss_once = np.dot(test_loss_vector[nc_id], num_test[nc_id]) / np.sum(
                    num_test[nc_id])

                malicious_train_acc_once = np.sum(num_correct_train[c_id]) * 1.0 / np.sum(num_train[c_id])
                malicious_test_acc_once = np.sum(num_correct_test[c_id]) * 1.0 / np.sum(num_test[c_id])
                benign_train_acc_once = np.sum(num_correct_train[nc_id]) * 1.0 / np.sum(num_train[nc_id])
                benign_test_acc_once = np.sum(num_correct_test[nc_id]) * 1.0 / np.sum(num_test[nc_id])
                var_of_performance_once = np.var(num_correct_test[nc_id] / num_test[nc_id])

                benign_test_acc.append(benign_test_acc_once)
                benign_train_acc.append(benign_train_acc_once)
                var_of_performance.append(var_of_performance_once)
                benign_test_loss.append(benign_test_loss_once)
                benign_train_loss.append(benign_train_loss_once)

                tqdm.write(
                    'At round {} malicious training accu: {}, loss: {}'.format(i, malicious_train_acc_once,
                                                                                        malicious_train_loss_once))
                tqdm.write(
                    'At round {} malicious test accu: {}, loss: {}'.format(i, malicious_test_acc_once,
                                                                                    malicious_test_loss_once))
                tqdm.write('At round {} benign training accu: {}, loss: {}'.format(i, benign_train_acc_once,
                                                                                            benign_train_loss_once))
                tqdm.write(
                    'At round {} benign test accu: {}, loss: {}'.format(i, benign_test_acc_once,
                                                                                 benign_test_loss_once))

                tqdm.write("At round {} variance of the performance: {}".format(i, var_of_performance_once))  # fairness

                if i == self.initial_rounds:
                    break
            indices, selected_clients = self.select_clients(round=i, num_clients=clients_per_round)
            print("initial_round {} selected clients indices: {}".format(i, indices))
            csolns = []

            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for epoch_i in range(self.epoch):
                    num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                    train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
                    random_seed += 1
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
                
                if idx in corrupt_id and (self.attack_type == 'A3' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 3)):
                    diff = [10 * u for u in diff]

                elif idx in corrupt_id and (self.attack_type == 'A2' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 2)):
                    stdev_ = get_stdev(diff)
                    diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                elif idx in corrupt_id and (self.attack_type == 'A4' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 4)):
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
            elif self.Robust ==5:
                m = clients_per_round - expected_num_mali
                avg_updates = self.mkrum_average(clients_per_round - expected_num_mali - 2, m, csolns)
            else:
                avg_updates = self.simple_average(csolns)

            # update the global model
            for layer in range(len(avg_updates)):
                self.global_model[layer] += avg_updates[layer]
                
        print("initial rounds is ending")
        indices = np.arange(len(self.clients))
        csolns = []

        #cluster
        if self.num_of_clusters != 1:
            for idx in indices:
                w_global_idx = copy.deepcopy(self.global_model)
                c = self.clients[idx]
                for epoch_i in range(self.epoch):
                    num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                    train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
                    random_seed += 1
                    for s in range(num_batch):
                        batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                        batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                        data_batch = (batch_xs, batch_ys)
                        # global
                        self.client_model.set_params(w_global_idx)
                        #loss = c.get_loss()
                        _, grads, _ = c.solve_sgd(data_batch)
                        w_global_idx = self.client_model.get_params()

                # get the difference (global model updates)
                diff = [u - v for (u, v) in zip(w_global_idx, self.global_model)]

                # send the malicious updates
                if idx in corrupt_id and (self.attack_type == 'A3' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 3)):
                    diff = [10 * u for u in diff]

                elif idx in corrupt_id and (self.attack_type == 'A2' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 2)):
                    stdev_ = get_stdev(diff)
                    diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                elif idx in corrupt_id and (self.attack_type == 'A4' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 4)):
                    diff = [-1 * u for u in diff]

                csolns.append(diff)
                
            G, Z = self.hierarchical_cluster(csolns, indices)
        else:
            G = [[i for i in range(len(self.clients))]]
            Z = []
        print("Groups info: {}".format(G))

        #extend train in each group using MTL
        Groups_details={}
        CM = []
        for g in range(len(G)):
            details = []
            group_c = [] #include self.clients info in each group

            for g_usr in G[g]:
                group_c.append(self.clients[g_usr])

            g_train_acc = []
            g_train_loss = []
            g_test_acc = []
            g_test_loss = []
            g_benign_train_acc = []
            g_benign_train_loss = []
            g_benign_test_acc = []
            g_benign_test_loss = []
            g_malicious_train_acc = []
            g_malicious_train_loss = []
            g_malicious_test_acc = []
            g_malicious_test_loss = []
            g_var_of_performance = []
            print("Now {}th group in G starts training".format(g))
            self.group_model = copy.deepcopy(self.global_model)

            g_clients_per_round = math.ceil(len(G[g]) * self.participation_ratio)
            print("Random select {} user(s) in {}th group".format(g_clients_per_round, g))

            for i in range(self.remain_rounds + 1):
                if i % self.eval_every == 0 and i > 0:
                    tmp_models = []
                    for idx in G[g]:
                        tmp_models.append(self.local_models[idx])

                    num_train, num_correct_train, train_loss_vector = self.train_error(tmp_models, group_c)
                    num_test, num_correct_test, test_loss_vector, Group_cm = self.test(tmp_models, group_c)
                    # if val
                    # num_val, num_correct_val, val_loss_vector, Group_cm = self.val(tmp_models, group_c)
                    if i == self.remain_rounds: # Group_cm.shape == (num_users_in_Group, (num_classes, num_classes))
                        CM.append(Group_cm)
                    train_acc_once = np.sum(num_correct_train) * 1.0 / np.sum(num_train)
                    avg_train_loss = np.dot(train_loss_vector, num_train) / np.sum(num_train)

                    test_acc_once = np.sum(num_correct_test) * 1.0 / np.sum(num_test)
                    avg_test_loss = np.dot(test_loss_vector, num_test) / np.sum(num_test)

                    tqdm.write('At round {} group {} training accu: {}, loss: {}'.format(i, g, train_acc_once, avg_train_loss))
                    tqdm.write('At round {} group {} test accu: {}, loss: {}'.format(i, g, test_acc_once, avg_test_loss))

                    c_id = []
                    nc_id = []

                    for idx, j in enumerate(G[g]):
                        if j in corrupt_id:
                            c_id.append(idx)
                        else:
                            nc_id.append(idx)
                    if nc_id == []:
                        break
                    malicious_train_loss_once = np.dot(train_loss_vector[c_id], num_train[c_id]) / np.sum(
                        num_train[c_id])
                    benign_train_loss_once = np.dot(train_loss_vector[nc_id], num_train[nc_id]) / np.sum(
                        num_train[nc_id])
                    malicious_test_loss_once = np.dot(test_loss_vector[c_id], num_test[c_id]) / np.sum(
                        num_test[c_id])
                    benign_test_loss_once = np.dot(test_loss_vector[nc_id], num_test[nc_id]) / np.sum(
                        num_test[nc_id])

                    malicious_train_acc_once = np.sum(num_correct_train[c_id]) * 1.0 / np.sum(num_train[c_id])
                    malicious_test_acc_once = np.sum(num_correct_test[c_id]) * 1.0 / np.sum(num_test[c_id])
                    benign_train_acc_once = np.sum(num_correct_train[nc_id]) * 1.0 / np.sum(num_train[nc_id])
                    benign_test_acc_once = np.sum(num_correct_test[nc_id]) * 1.0 / np.sum(num_test[nc_id])
                    var_of_performance_once = np.var(num_correct_test[nc_id] / num_test[nc_id])

                    usr_in_group_acc = []
                    for user_id, user_acc in zip(np.array(G[g])[np.array(nc_id)], num_correct_test[nc_id]/num_test[nc_id]):
                        usr_in_group_acc.append((user_id, user_acc))


                    details.append(("round"+str(i),
                                    num_correct_test[nc_id],
                                    num_test[nc_id],
                                    num_correct_test[nc_id] / num_test[nc_id],
                                    num_correct_train[nc_id],
                                    num_train[nc_id],
                                    num_correct_train[nc_id] / num_train[nc_id],
                                    train_loss_vector[nc_id],
                                    test_loss_vector[nc_id]))


                    tqdm.write('At round {} group {} malicious training accu: {}, loss: {}'.format(i, g, malicious_train_acc_once,
                                                                                          malicious_train_loss_once))
                    tqdm.write('At round {} group {} malicious test accu: {}, loss: {}'.format(i, g, malicious_test_acc_once,
                                                                                      malicious_test_loss_once))
                    tqdm.write('At round {} group {} benign training accu: {}, loss: {}'.format(i, g, benign_train_acc_once,
                                                                                       benign_train_loss_once))
                    tqdm.write(
                        'At round {} group {} benign test accu: {}, loss: {}'.format(i, g, benign_test_acc_once, benign_test_loss_once))
                    tqdm.write('At round {} users in group {} benign test accu: {}'.format(i, g, usr_in_group_acc))
                    tqdm.write("group {} variance of the performance: {}".format(g, var_of_performance_once))  # fairness

                    g_train_acc.append(train_acc_once)
                    g_train_loss.append(avg_train_loss)
                    g_test_acc.append(test_acc_once)
                    g_test_loss.append(avg_test_loss)

                    g_benign_train_acc.append(benign_train_acc_once)
                    g_benign_test_acc.append(benign_test_acc_once)
                    g_benign_train_loss.append(benign_train_loss_once)
                    g_benign_test_loss.append(benign_test_loss_once)

                    g_malicious_train_acc.append(malicious_train_acc_once)
                    g_malicious_test_acc.append(malicious_test_acc_once)
                    g_malicious_train_loss.append(malicious_train_loss_once)
                    g_malicious_test_loss.append(malicious_test_loss_once)
                    g_var_of_performance.append(var_of_performance_once)
                    if i == self.remain_rounds:
                        break
                indices, selected_clients = self.select_clients_intraG(i + self.initial_rounds, g_clients_per_round, G[g], group_c)
                print('Round {} select user(s) {} in group {}'.format(i, indices, g))
                csolns = []

                for idx in indices:
                    w_global_idx = copy.deepcopy(self.group_model)
                    c = self.clients[idx]
                    for epoch_i in range(self.epoch):
                        num_batch = int(math.ceil(len(c.train_data['y']) / self.batch_size))
                        train_x, train_y = self.shuffle_training_data(c.train_data['x'], c.train_data['y'], random_seed)
                        random_seed += 1
                        for s in range(num_batch):
                            batch_xs = train_x[s * self.batch_size: (s + 1) * self.batch_size]
                            batch_ys = train_y[s * self.batch_size: (s + 1) * self.batch_size]
                            data_batch = (batch_xs, batch_ys)
                            # local
                            self.client_model.set_params(self.local_models[idx])
                            _, grads, _ = c.solve_sgd(data_batch)  # weights,grads,loss

                            if self.dynamic_lamda:

                                model_tmp = copy.deepcopy(self.local_models[idx])
                                model_best = copy.deepcopy(self.local_models[idx])
                                tmp_loss = 10000
                                # pick a lambda locally based on validation data
                                for lam_id, candidate_lam in enumerate([0.1, 1, 2]):
                                    for layer in range(len(grads[1])):
                                        eff_grad = grads[1][layer] + candidate_lam * (self.local_models[idx][layer] - self.group_model[layer])
                                        model_tmp[layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                                    c.set_params(model_tmp)
                                    l = c.get_val_loss()
                                    if l < tmp_loss:
                                        tmp_loss = l
                                        model_best = copy.deepcopy(model_tmp)

                                self.local_models[idx] = copy.deepcopy(model_best)

                            else:
                                for layer in range(len(grads[1])):
                                    eff_grad = grads[1][layer] + self.lamda * (self.local_models[idx][layer] - self.group_model[layer])  # vk= vk− η(∇Fk(vk) + λ(vk− wt))
                                    self.local_models[idx][layer] = self.local_models[idx][layer] - self.learning_rate * eff_grad

                            # global
                            self.client_model.set_params(w_global_idx)
                            # loss = c.get_loss()
                            _, grads, _ = c.solve_sgd(data_batch)
                            w_global_idx = self.client_model.get_params()

                    # get the difference (global model updates)
                    diff = [u - v for (u, v) in zip(w_global_idx, self.group_model)]

                    # send the malicious updates
                    if idx in corrupt_id and (self.attack_type == 'A3' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 3)):
                        diff = [10 * u for u in diff]

                    elif idx in corrupt_id and (self.attack_type == 'A2' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 2)):
                        stdev_ = get_stdev(diff)
                        diff = [np.random.normal(0, stdev_, size=u.shape) for u in diff]

                    elif idx in corrupt_id and (self.attack_type == 'A4' or (self.attack_type == 'Hybrid' and attack_belong[list(corrupt_id).index(idx)] == 4)):
                        diff = [-1 * u for u in diff]

                    csolns.append(diff)

                if self.Robust == 1:
                    csolns = l2_clip(csolns)

                g_expected_num_mali = int(g_clients_per_round * self.corrupted_ratio)

                if self.Robust == 2:
                    avg_updates = self.median_average(csolns)
                elif self.Robust == 3:
                    avg_updates = self.k_norm_average(g_clients_per_round - g_expected_num_mali, csolns)
                elif self.Robust == 4:
                    avg_updates = self.krum_average(g_clients_per_round - g_expected_num_mali - 2, csolns)
                elif self.Robust == 5:
                    m = g_clients_per_round - g_expected_num_mali
                    avg_updates = self.mkrum_average(g_clients_per_round - g_expected_num_mali - 2, m, csolns)
                else:
                    avg_updates = self.simple_average(csolns)

                # update the global model
                for layer in range(len(avg_updates)):
                    self.group_model[layer] += avg_updates[layer]

            Groups_details["Group" + str(g)] = details


            print("group {} train_acc:{}".format(g, g_train_acc))
            print("group {} train_loss:{}".format(g, g_train_loss))
            print("group {} test_acc:{}".format(g, g_test_acc))
            print("group {} test_loss:{}".format(g, g_test_loss))
            print("group {} benign_train_acc:{}".format(g, g_benign_train_acc))
            print("group {} benign_train_loss:{}".format(g, g_benign_train_loss))
            print("group {} benign_test_acc:{}".format(g, g_benign_test_acc))
            print("group {} benign_test_loss:{}".format(g, g_benign_test_loss))
            print("group {} malicious_train_acc:{}".format(g, g_malicious_train_acc))
            print("group {} malicious_test_acc:{}".format(g, g_malicious_test_acc))
            print("group {} malicious_train_loss:{}".format(g, g_malicious_train_loss))
            print("group {} malicious_test_loss:{}".format(g, g_malicious_test_loss))
            print("group {} var_of_performance:{}".format(g, g_var_of_performance))
            print("Now {}th group in G ends training".format(g))

        print("**********************************")
        # print("CM:\n", CM)

        for eval_r in range(self.remain_rounds):
            all_test_num_correct = np.array([])
            all_num_test = np.array([])
            all_train_num_correct = np.array([])
            all_num_train = np.array([])
            test_acc_vec = np.array([])
            train_acc_vec = np.array([])
            test_loss_vec = np.array([])
            train_loss_vec = np.array([])
            for g in range(len(G)):
                if Groups_details["Group"+str(g)] == []:
                    continue
                all_test_num_correct = np.append(all_test_num_correct, Groups_details["Group"+str(g)][eval_r][1])
                all_num_test = np.append(all_num_test, Groups_details["Group"+str(g)][eval_r][2])
                test_acc_vec = np.append(test_acc_vec, Groups_details["Group"+str(g)][eval_r][3])


                all_train_num_correct = np.append(all_train_num_correct, Groups_details["Group"+str(g)][eval_r][4])
                all_num_train = np.append(all_num_train, Groups_details["Group"+str(g)][eval_r][5])
                train_acc_vec = np.append(train_acc_vec, Groups_details["Group" + str(g)][eval_r][6])

                train_loss_vec = np.append(train_loss_vec, Groups_details["Group" + str(g)][eval_r][7])
                test_loss_vec = np.append(test_loss_vec, Groups_details["Group"+str(g)][eval_r][8])


            benign_test_acc.append(np.sum(all_test_num_correct) / np.sum(all_num_test))
            benign_train_acc.append(np.sum(all_train_num_correct) / np.sum(all_num_train))
            var_of_performance.append(np.var(test_acc_vec))
            benign_test_loss.append(np.dot(train_loss_vec, all_num_test) / np.sum(all_num_test))
            benign_train_loss.append(np.dot(test_loss_vec, all_num_train) / np.sum(all_num_train))


        nc_id = []
        if self.attack_type != 'B':
            for g in range(len(G)):
                for j in G[g]:
                    if j not in corrupt_id:
                        nc_id.append(j)
        else:
            nc_id = list(np.arange(len(self.clients)))

        final_benign_train_acc_per_user = {}
        final_benign_test_acc_per_user = {}

        for i, j in zip(nc_id, list(train_acc_vec)):
            final_benign_train_acc_per_user['{}'.format(i)] = j
        for i, j in zip(nc_id, list(test_acc_vec)):
            final_benign_test_acc_per_user['{}'.format(i)] = j

        print("Last round benign train acc per user: {}".format(final_benign_train_acc_per_user))
        print("benign_train_acc: {}".format(benign_train_acc))
        print("benign_train_loss: {}".format(benign_train_loss))
        print("Last round benign test acc per user: {}".format(final_benign_test_acc_per_user))
        print("benign_test_acc: {}".format(benign_test_acc))
        print("benign_test_loss: {}".format(benign_test_loss))
        print("var_of_performance: {}".format(var_of_performance))




        #process CM
        # CM  == [cm,cm,cm,cm] -> cm == array:(num_users_in_Group, num_classes, num_classes)
        CMN = np.zeros([self.num_classes, self.num_classes])
        for i in range(len(CM)):
            for j in range(CM[i].shape[0]):
                CMN += CM[i][j]
        print("[NC]Confusion Matrix:\n ", CMN)
        CMR = CMN.astype('float') / CMN.sum(axis=1)[:, np.newaxis]
        CMR = np.around(CMR, decimals=2)
        print("[ACC]Confusion Matrix:\n ", CMR)

        if not os.path.exists('./record/Dataset/{}'.format(self.dataset)):
            os.makedirs('./record/Dataset/{}'.format(self.dataset))

        np.savez('./record/Dataset/{}/FedCHAR_ir{}_rr{}_c{}_pr{}_atk{}_cr{}_lr{}_bz{}_ep{}_R{}_lkg{}_lamda{}.npz'.format(self.dataset,
                                                                                         self.initial_rounds,
                                                                                         self.remain_rounds,
                                                                                         self.num_of_clusters,
                                                                                         self.participation_ratio,
                                                                                         self.attack_type,
                                                                                         self.corrupted_ratio,
                                                                                         self.learning_rate,
                                                                                         self.batch_size,
                                                                                         self.epoch,
                                                                                         self.Robust,
                                                                                         self.linkage,
                                                                                         self.lamda),
                 final_benign_train_acc_per_user=final_benign_train_acc_per_user,
                 benign_train_acc=benign_train_acc,
                 benign_train_loss=benign_train_loss,
                 final_benign_test_acc_per_user=final_benign_test_acc_per_user,
                 benign_test_acc=benign_test_acc,
                 benign_test_loss=benign_test_loss,
                 var_of_performance=var_of_performance,
                 CMN=CMN,
                 CMR=CMR,
                 Z=Z,
                 G=G,
                 param=self.params,
                 attack_belong=attack_belong_dict)

