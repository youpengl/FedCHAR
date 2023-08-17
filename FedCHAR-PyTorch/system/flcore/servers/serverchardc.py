import os
import copy
import numpy as np
from flcore.clients.clientchardc import clientCHARDC
from flcore.servers.serverbase import Server
from utils.byzantine import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

class FedCHAR_DC(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientCHARDC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

        self.initial_rounds = args.initial_rounds
        self.n_clusters = args.n_clusters
        self.metric = args.metric
        self.linkage = args.linkage
        self.recluster_rounds = args.recluster_rounds
        self.initial_group_updates = []

    def train(self):

        '''Initial Stage'''
        for i in range(self.initial_rounds):
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models()

            if self.future_test:
                print("\nEvaluate group models for future clients.")
                selected_clients = self.selected_clients
                self.selected_clients = [client for client in self.clients if client.id not in self.training_clients_ids]
                for client in self.selected_clients:
                    client.set_parameters(self.global_model)
                self.evaluate_for_future_clients()
                self.selected_clients = selected_clients
                
            for client in self.selected_clients:
                client.dtrain()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models for training clients.")
                self.evaluate_personalized()

            self.receive_models()
            self.aggregate_parameters()

        '''Clustering Stage'''
        self.current_round += 1
        self.selected_clients = self.select_clients()
        self.send_models()

        if self.future_test:
            print("\nEvaluate group models for future clients.")
            selected_clients = self.selected_clients
            self.selected_clients = [client for client in self.clients if client.id not in self.training_clients_ids]
            for client in self.selected_clients:
                client.set_parameters(self.global_model)
            self.evaluate_for_future_clients()
            self.selected_clients = selected_clients

        for client in self.selected_clients:
            client.dtrain()
        
        print(f"\n-------------Clustering round: {self.initial_rounds}-------------")
        print("\n-Evaluate fine-tuned models for training clients ")
        self.evaluate_personalized()

        self.receive_models('Cluster')
        client_updates = [torch.cat([uu.reshape(-1, 1) for uu in u], axis=0).detach().cpu().numpy().squeeze() for u in self.uploaded_updates]
        self.cluster_identity = self.cluster(client_updates)
        cluster_info = [[('Malicious' if self.selected_clients[idx].malicious else 'Benign', self.uploaded_ids[idx]) for idx, g_id in enumerate(self.cluster_identity) if g_id == i] for i in range(max(self.cluster_identity+1))]
        for idx, info in enumerate(cluster_info):
            print('Cluster {}: {}'.format(idx, info))
        self.group_models = [copy.deepcopy(self.global_model)] * (max(self.cluster_identity) + 1)

        self.user_belong_g = np.ones([len(self.training_clients)]).astype('int') * -1
        for u_idx, u_id in enumerate(self.uploaded_ids):
            u_relative_id = list(self.training_clients_ids).index(u_id)
            self.user_belong_g[u_relative_id] = self.cluster_identity[u_idx]

        self.aggregate_parameters_g()
        
        '''Remaining Stage'''
        self.state = np.zeros([len(self.training_clients)]).astype(int)
        for i in range(self.global_rounds - (self.initial_rounds + 1)):
            self.current_round += 1
            self.selected_clients = self.select_clients()
            self.send_models_g()
            self.state[[list(self.training_clients_ids).index(c.id) for c in self.training_clients if c not in self.selected_clients]] += 1
            self.state[[list(self.training_clients_ids).index(c.id) for c in self.selected_clients]] = 0

            if self.future_test:
                print("\nEvaluate group models for future clients.")
                selected_clients = self.selected_clients
                self.selected_clients = [client for client in self.clients if client.id not in self.training_clients_ids]
                for client in self.selected_clients:
                    client_update = client.get_update(self.global_model)
                    g_idx = self.caculate_similarity(client_update, self.initial_group_updates)
                    if g_idx == len(self.group_models):
                        client.set_parameters(self.global_model)
                    else:
                        client.set_parameters(self.group_models[g_idx])
                self.evaluate_for_future_clients()
                self.selected_clients = selected_clients
                
            for client in self.selected_clients:
                client.dtrain()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i+self.initial_rounds+1}-------------")
                print("\nEvaluate personalized models for training clients.")
                self.evaluate_personalized()

            self.receive_models_g()
            self.aggregate_parameters_g()

        print(f"\n-------------Final Report-------------")
        print("\nFinal Average Personalized Accuracy: {:.2f}%\nFinal Average Future Accuracy: {:.2f}%\n".format(self.rs_test_acc_p[-1]*100, self.ft_test_acc[-1]*100 if self.ft_test_acc != [] else -1))
        self.save_global_model()

    def receive_models(self, stage=None):

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_updates = []

        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_updates.append([c_param.data - s_param.data for c_param, s_param in zip(client.model.parameters(), self.global_model.parameters())])
    
        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(self.uploaded_updates, malicious_ids)

        if stage != 'Cluster':
            for i, w in enumerate(self.uploaded_weights):
                self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_updates) > 0)

        self.global_update = copy.deepcopy(self.uploaded_updates[0])
        for param in self.global_update:
            param.data.zero_()
            
        for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
            self.add_parameters(w, client_update)

        for model_param, update_param in zip(self.global_model.parameters(), self.global_update):
            model_param.data += update_param.data.clone()

    def add_parameters(self, w, client_update):
        for server_param, client_param in zip(self.global_update, client_update):
            server_param.data += client_param.data.clone() * w

    def collect(self):
        clients_updates = []
        for client in self.selected_clients:
            client.ptrain()
            client.train()
            clients_updates.append(client.get_update(self.global_model))

        if self.attack_type != 'B' and self.attack_type != 'A1':
            clients_updates = eval(self.attack_type)(clients_updates, self.malicious_ids.tolist(), len(self.selected_clients))
        
        clients_updates = [torch.cat([uu.reshape(-1, 1) for uu in u], axis=0).detach().cpu().numpy().squeeze() for u in clients_updates]
        return clients_updates
    
    def cluster(self, clients_updates):
        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.metric, linkage=self.linkage).fit(clients_updates)
        return clustering.labels_

    def aggregate_parameters_g(self):
        
        for i in range(len(self.group_models)):
            self.global_update = copy.deepcopy(self.uploaded_updates[0])
            for param in self.global_update:
                param.data.zero_()
            
            uploaded_weights = [self.uploaded_weights[u_id] for u_id in range(len(self.uploaded_weights)) if self.user_belong_g[list(self.training_clients_ids).index(self.uploaded_ids[u_id])] == i]
            uploaded_weights = [weight / sum(uploaded_weights) for weight in uploaded_weights]
            uploaded_updates = [self.uploaded_updates[u_id] for u_id in range(len(self.uploaded_updates)) if self.user_belong_g[list(self.training_clients_ids).index(self.uploaded_ids[u_id])] == i]

            for w, client_update in zip(uploaded_weights, uploaded_updates):
                self.add_parameters(w, client_update)

            if len(self.initial_group_updates) != len(self.group_models):
                self.initial_group_updates.append(self.global_update)

            for model_param, update_param in zip(self.group_models[i].parameters(), self.global_update):
                model_param.data += update_param.data.clone()

    def send_models_g(self):
        for client in self.selected_clients:
            c_id = list(self.training_clients_ids).index(client.id)
            if self.user_belong_g[c_id] != -1 and self.state[c_id] < self.recluster_rounds:
                client.set_parameters(self.group_models[self.user_belong_g[c_id]])
            else:
                client_update = client.get_update(self.global_model)
                if self.attack_type != 'B' and self.attack_type != 'A1':
                    client_update = eval(self.attack_type)(client_update, None, len(self.selected_clients))
                g_idx = self.caculate_similarity(client_update, self.initial_group_updates)
                if g_idx == len(self.group_models):
                    new_group_model = copy.deepcopy(self.global_model)
                    self.group_models.append(new_group_model)
                    self.initial_group_updates.append(client_update)
                    client.set_parameters(new_group_model)
                else:
                    client.set_parameters(self.group_models[g_idx])

                self.user_belong_g[c_id] = g_idx

    def caculate_similarity(self, client_update, group_updates): 
        max = -1
        min = 999
        g_belong = -1
        '''In our paper, the shape is equal to the sum of dimensions of all layers.
        Here, we only calculate the similarity between last layers of models to solve the memory issue.'''
        shape = group_updates[0][-2].shape[0] * group_updates[0][-2].shape[1]
        for g_idx, csolns_i in enumerate(group_updates):
            for g_jdx, csolns_j in enumerate(group_updates):
                if g_idx < g_jdx:
                    v1 = torch.cat([u.reshape(1, -1) for u in csolns_i], axis=1).detach().cpu().numpy()[-shape:]
                    v2 = torch.cat([u.reshape(1, -1) for u in csolns_j], axis=1).detach().cpu().numpy()[-shape:]
                    cos_sim = cosine_similarity(v1, v2)[0][0]

                    if cos_sim < min:
                        min = cos_sim

        main = torch.cat([u.reshape(1, -1) for u in client_update], axis=1).detach().cpu().numpy()[-shape:]
        for g_idx, csolns in enumerate(group_updates):
            other = torch.cat([u.reshape(1, -1) for u in csolns], axis=1).detach().cpu().numpy()[-shape:]
            cos_sim = cosine_similarity(main, other)[0][0]
            if cos_sim > max:
                max = cos_sim
                g_belong = g_idx

        if len(group_updates) == 1: # Only one cluster
            if max > 0.:
                return g_belong
            else:
                print('Cosine similarity between new user and the cluster: {}'.format(max))
                return len(group_updates)

        else:
            if max < min:
                print('Min cosine similarity among existing groups: {}'.format(min))
                print('Max cosine similarity between new user and existing groups: {}'.format(max))
                return len(group_updates)
            else:
                return g_belong
            
    def receive_models_g(self):
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_updates = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            c_id = list(self.training_clients_ids).index(client.id)
            self.uploaded_updates.append([c_param.data - s_param.data for c_param, s_param in zip(client.model.parameters(), self.group_models[self.user_belong_g[c_id]].parameters())])
    
        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(self.uploaded_updates, malicious_ids)
            
    def func_future_test(self, model_path=None):
        self.selected_clients = [client for client in self.clients if client.id not in self.training_clients_ids]
        if model_path == None:
            checkpoint = torch.load(self.model_path)
        else:
            checkpoint = torch.load(model_path)
        global_model = checkpoint['global_model']
        group_models = checkpoint['group_models']
        initial_group_updates = checkpoint['initial_group_updates']

        for client in self.selected_clients:

            client_update = client.get_update(global_model)
            g_idx = self.caculate_similarity(client_update, initial_group_updates)
            if g_idx == len(self.group_models):
                client.set_parameters(global_model)
            else:
                client.set_parameters(group_models[g_idx])


        print("\n--------Evaluate initial model--------")
        self.evaluate_for_future_clients()

        print("\n--------Evaluate fine-tuning models--------")
        for i in range(self.finetune_rounds):
            print(f"\n-------------Fine-tuning round number: {i}-------------")
            
            for client in self.selected_clients:
                client.fine_tuning()

            self.evaluate_for_future_clients()

        print(f"\n-------------Final Report for Future Clients-------------")
        print("\nInitial Average Accuracy for Future Clients: {:.2f}\nFinal Average Personalized Accuracy: {:.2f}\n".format(self.ft_test_acc[self.global_rounds+1], self.ft_test_acc[-1]))

    def save_global_model(self):
        model_path ="models/"

        filename = "{}_{}_{}_atkR{}_gr{}_ep{}_bz{}_lr{}_nc{}_fur{}_ntc{}_jr{}_ftr{}_seed{}".format(self.dataset, self.algorithm, self.attack_type, self.attack_ratio,
                                                                            self.global_rounds, self.local_steps, self.batch_size, self.learning_rate,
                                                                            self.num_clients, self.future_ratio, self.num_training_clients, self.join_ratio, self.finetune_rounds, self.seed)


        filename = filename + '_ir{}_ng{}_mtrc{}_lkg{}_rr{}'.format(self.initial_rounds, self.n_clusters, self.metric, self.linkage, self.recluster_rounds)
        
        filename = filename + '.tar'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        self.model_path = os.path.join(model_path, filename)

        print("Model path: " + self.model_path)

        torch.save({'global_model': self.global_model,
                    'group_models': self.group_models,
                    'initial_group_updates': self.initial_group_updates,
                    'user_belong_g': self.user_belong_g}, self.model_path)