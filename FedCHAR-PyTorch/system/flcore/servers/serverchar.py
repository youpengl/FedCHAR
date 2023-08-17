import copy
import numpy as np
from flcore.clients.clientchar import clientCHAR
from flcore.servers.serverbase import Server
from utils.byzantine import *
from sklearn.cluster import AgglomerativeClustering

class FedCHAR(Server):
    def __init__(self, args):
        super().__init__(args)

        self.set_clients(args, clientCHAR)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_training_clients}")
        print("Finished creating server and clients.")

        self.initial_rounds = args.initial_rounds
        self.n_clusters = args.n_clusters
        self.metric = args.metric
        self.linkage = args.linkage
        

    def train(self):
        # initial Stage
        for i in range(self.initial_rounds):
            self.selected_clients = self.select_clients()
            self.send_models()
                
            for client in self.selected_clients:
                client.dtrain()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models for training clients.")
                self.evaluate_personalized()

            self.receive_models()
            self.aggregate_parameters()

        # Clustering Stage
        print(f"\n-------------Clustering-------------")
        clients_updates = self.collect()
        self.cluster_identity = self.cluster(clients_updates)
        cluster_info = [[('Malicious' if self.training_clients[idx].malicious else 'Benign', idx) for idx, g_id in enumerate(self.cluster_identity) if g_id == i] for i in range(max(self.cluster_identity)+1)]
        for idx, info in enumerate(cluster_info):
            print('Cluster {}: {}'.format(idx, info))

        self.group_models = [copy.deepcopy(self.global_model)] * (max(self.cluster_identity) + 1)

        # Remaining Stage
        for i in range(self.global_rounds - self.initial_rounds):
            self.selected_clients = self.select_clients()
            self.send_models_g()

            for client in self.selected_clients:
                client.dtrain()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i+self.initial_rounds}-------------")
                print("\nEvaluate personalized models for training clients.")
                self.evaluate_personalized()

            self.receive_models_g()
            self.aggregate_parameters_g()

        print("\nFinal Average Personalized Accuracy: {}\n".format(self.rs_test_acc_p[-1]))

    def receive_models(self):

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

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def add_parameters(self, w, client_update):

        for server_param, client_param in zip(self.global_update, client_update):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):

        self.global_update = copy.deepcopy(self.uploaded_updates[0])
        for param in self.global_update:
            param.data.zero_()
            
        for w, client_update in zip(self.uploaded_weights, self.uploaded_updates):
            self.add_parameters(w, client_update)

        for model_param, update_param in zip(self.global_model.parameters(), self.global_update):
            model_param.data += update_param.data.clone()

    def collect(self):
        
        clients_updates = []
        for client in self.training_clients:
            clients_updates.append(client.get_update(self.global_model))

        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.training_clients_ids) if c_id in self.malicious_ids]
            clients_updates = eval(self.attack_type)(clients_updates, malicious_ids, len(self.selected_clients))
        
        clients_updates = [torch.cat([uu.reshape(-1, 1) for uu in u], axis=0).detach().cpu().numpy().squeeze() for u in clients_updates]
        return clients_updates
    
    def cluster(self, clients_updates):

        clustering = AgglomerativeClustering(n_clusters=self.n_clusters, metric=self.metric, linkage=self.linkage).fit(clients_updates)
        return clustering.labels_

    def send_models_g(self):

        for client in self.selected_clients:  
            c_idx = list(self.training_clients_ids).index(client.id)
            client.set_parameters(self.group_models[self.cluster_identity[c_idx]])

    def receive_models_g(self):

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_updates = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            c_idx = list(self.training_clients_ids).index(client.id)
            self.uploaded_updates.append([c_param.data - s_param.data for c_param, s_param in zip(client.model.parameters(), self.group_models[self.cluster_identity[c_idx]].parameters())])
    
        if self.attack_type != 'B' and self.attack_type != 'A1':
            malicious_ids = [idx for idx, c_id in enumerate(self.uploaded_ids) if c_id in self.malicious_ids]
            self.uploaded_updates = eval(self.attack_type)(self.uploaded_updates, malicious_ids)

    def aggregate_parameters_g(self):

        for i in range(len(self.group_models)):
            self.global_update = copy.deepcopy(self.uploaded_updates[0])
            for param in self.global_update:
                param.data.zero_()
            
            user_idx_in_same_group = np.array([r_id for r_id, c_id in enumerate(self.uploaded_ids) if self.cluster_identity[list(self.training_clients_ids).index(c_id)] == i])
            uploaded_weights = [self.uploaded_weights[u_id] for u_id in range(len(self.uploaded_weights)) if u_id in user_idx_in_same_group]
            uploaded_weights = [weight / sum(uploaded_weights) for weight in uploaded_weights]
            uploaded_updates = [self.uploaded_updates[u_id] for u_id in range(len(self.uploaded_updates)) if u_id in user_idx_in_same_group]

            for w, client_update in zip(uploaded_weights, uploaded_updates):
                self.add_parameters(w, client_update)

            for model_param, update_param in zip(self.group_models[i].parameters(), self.global_update):
                model_param.data += update_param.data.clone()

