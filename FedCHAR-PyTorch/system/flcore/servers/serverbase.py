import torch
import os
import numpy as np
import copy
import wandb 
from utils.byzantine import *

class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.attack_ratio = args.attack_ratio
        self.attack_type = args.attack_type
        self.seed = args.seed
        self.algorithm = args.algorithm
        self.args = args
        self.current_round = -1
        self.future_test = args.future_test
        self.future_ratio = args.future_ratio
        self.num_training_clients = args.num_clients - int(args.num_clients * args.future_ratio)
        self.join_clients = int(self.num_training_clients * self.join_ratio)
        self.finetune_rounds = args.finetune_rounds
        self.eval_gap = args.eval_gap
        self.detailed_info = args.detailed_info
        self.partition = args.partition
        self.data_path = args.data_path

        self.clients = []
        self.training_clients = []
        self.malicious_ids = []
        self.selected_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        self.uploaded_updates = []
        
        self.rs_test_acc_g = []
        self.rs_train_loss_g = []
        self.rs_test_accs_g = []
        self.rs_test_acc_p = []
        self.rs_train_loss_p = []
        self.rs_test_accs_p = []
        self.ft_train_loss = []
        self.ft_test_acc = []
        self.ft_std_acc = []

    def set_clients(self, args, clientObj):

        if self.future_test == False:

            if self.attack_type == 'B':
                self.malicious_ids = []
                self.attack_ratio = 0.0
            else:
                self.malicious_ids = np.sort(np.random.choice(np.arange(self.num_clients), int(self.num_clients * self.attack_ratio), replace=False))
                

            for i in range(self.num_clients):
                        
                client = clientObj(args, 
                                id=i, 
                                malicious=True if i in self.malicious_ids else False)
                self.clients.append(client)
            
            self.training_clients = self.clients
            self.training_clients_ids = np.arange(self.num_clients)
        
        else:

            if self.algorithm != 'FedCHAR_DC':
                print('{} do not support future testing'.format(self.algorithm))
                raise NotImplementedError
            
            self.training_clients_ids = np.sort(np.random.choice(np.arange(self.num_clients), self.num_training_clients, replace=False))

            if self.attack_type == 'B':
                self.malicious_ids = []
                self.attack_ratio = 0.0
            else:
                self.malicious_ids = np.sort(np.random.choice(self.training_clients_ids, int(self.num_training_clients * self.attack_ratio), replace=False))

            for i in range(self.num_clients):
                        
                client = clientObj(args, 
                                id=i, 
                                malicious=True if i in self.malicious_ids else False)
                
                self.clients.append(client)

                if i in self.training_clients_ids:
                    self.training_clients.append(client)

        print('Malicious Clients: {}'.format(list(self.malicious_ids)))
        print('Future Clients: {}'.format(list(np.sort(np.setdiff1d(np.arange(self.num_clients), self.training_clients_ids)))))

    def select_clients(self):
        
        selected_clients = list(np.random.choice(self.training_clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):

        for client in self.selected_clients:
            client.set_parameters(self.global_model)

                
    def send_models_to_future_clients(self):

        for client in self.selected_clients:      
            client.set_parameters(self.global_model)

    def receive_models(self):

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []

        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
    
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):

        filename = "{}_{}_{}_{}_{}_bz{}_lr{}_gr{}_ep{}_jr{}_nc{}_fur{}_ntc{}_ftr{}_seed{}".format(self.dataset, self.partition, self.algorithm, 
                                                                                           self.attack_type, self.attack_ratio, self.batch_size, 
                                                                                           self.learning_rate, self.global_rounds, self.local_steps, 
                                                                                           self.join_ratio, self.num_clients, self.future_ratio, 
                                                                                           self.num_training_clients, self.finetune_rounds, self.seed)
        
        if self.algorithm == 'FedCHAR':
            filename = filename + '_ir{}_ng{}_mtrc{}_lkg{}'.format(self.initial_rounds, self.n_clusters, self.metric, self.linkage)

        elif self.algorithm == 'FedCHAR_DC':
            filename = filename + '_ir{}_ng{}_mtrc{}_lkg{}_rr{}'.format(self.initial_rounds, self.n_clusters, self.metric, self.linkage, self.recluster_rounds)

        result_path = "results/npz/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc_g) or len(self.rs_test_acc_p):
            file_path = result_path + "{}.npz".format(filename)
            print("Result path: " + file_path)

            np.savez(file_path, test_acc_g=self.rs_test_acc_g, 
                    test_acc_p=self.rs_test_acc_p, test_accs_g=self.rs_test_accs_g, 
                    test_accs_p=self.rs_test_accs_p, train_loss_g=self.rs_train_loss_g, 
                    train_loss_p=self.rs_train_loss_p, ft_train_loss=self.ft_train_loss, 
                    ft_test_acc=self.ft_test_acc, ft_std_acc=self.ft_std_acc)


    def test_metrics_for_future_clients(self):
        num_samples = []
        tot_correct = []
        
        for c in self.selected_clients:
            ct, ns = c.new_test_metrics()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, tot_correct

    def train_metrics_for_future_clients(self):
        num_samples = []
        losses = []
        for c in self.selected_clients:
            cl, ns = c.new_train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.selected_clients]

        return ids, num_samples, losses

    def evaluate_personalized(self, acc=None, loss=None):
        stats = self.test_metrics_personalized()
        stats_train = self.train_metrics_personalized()

        if self.malicious_ids != []:
            relative_malicious_ids = np.array([stats[0].index(i) for i in self.malicious_ids])

            stats_A = np.array(stats)[:, relative_malicious_ids].tolist()
            stats_train_A = np.array(stats_train)[:, relative_malicious_ids].tolist()

            test_acc_A = sum(stats_A[2])*1.0 / sum(stats_A[1])
            train_loss_A = sum(stats_train_A[2])*1.0 / sum(stats_train_A[1])
            accs_A = [a / n for a, n in zip(stats_A[2], stats_A[1])]
            losses_A = [a / n for a, n in zip(stats_train_A[2], stats_train_A[1])]

        else:
            test_acc_A = -1
            train_loss_A = -1
            accs_A = []
            losses_A = []


        benign_ids = np.sort(np.setdiff1d(self.training_clients_ids, self.malicious_ids))
        relative_benign_ids = np.array([stats[0].index(i) for i in benign_ids])

        stats_B = np.array(stats)[:, relative_benign_ids].tolist()
        stats_train_B = np.array(stats_train)[:, relative_benign_ids].tolist()

        stats = None
        stats_train = None

        test_acc = sum(stats_B[2])*1.0 / sum(stats_B[1])
        train_loss = sum(stats_train_B[2])*1.0 / sum(stats_train_B[1])
        accs = [a / n for a, n in zip(stats_B[2], stats_B[1])]
        losses = [a / n for a, n in zip(stats_train_B[2], stats_train_B[1])]

        
        if acc == None:
            self.rs_test_acc_p.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss_p.append(train_loss)
        else:
            loss.append(train_loss)

        self.rs_test_accs_p.append(accs)

        print("Benign Averaged Train Loss: {:.2f}".format(train_loss))
        print("Benign Averaged Test Accurancy: {:.2f}%".format(test_acc*100))
        print("Benign Std Test Accurancy: {:.2f}%".format(np.std(accs)*100))

        if self.malicious_ids != []:
            print("Malicious Averaged Train Loss: {:.2f}".format(train_loss_A))
            print("Malicious Averaged Test Accurancy: {:.2f}%".format(test_acc_A*100))

        try:
            if wandb.config['mode'] != 'debug':
                wandb.log({'p_train_loss':train_loss, 'p_test_acc':test_acc, 'p_std_test_acc':np.std(accs)})
        except Exception:
            pass
        
    def evaluate_for_future_clients(self):
        stats = self.test_metrics_for_future_clients()
        stats_train = self.train_metrics_for_future_clients()
        stats = np.array(stats).tolist()
        stats_train = np.array(stats_train).tolist()
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        losses = [a / n for a, n in zip(stats_train[2], stats_train[1])]

        print("Averaged Future Train Loss: {:.2f}".format(train_loss))
        print("Averaged Future Test Accurancy: {:.2f}%".format(test_acc*100))
        print("Std Future Test Accurancy: {:.2f}%".format(np.std(accs)*100))
        
        if self.detailed_info:
            print('Future Clients Train Loss:\n', [(int(stats[0][idx]), format(loss, '.2f')) for idx, loss in enumerate(losses)])
            print('Future Clients Test Accuracy:\n', [(int(stats[0][idx]), format(acc*100, '.2f')+'%') for idx, acc in enumerate(accs)])

        self.ft_train_loss.append(train_loss)
        self.ft_test_acc.append(test_acc)
        self.ft_std_acc.append(np.std(accs))

        try:
            if wandb.config['mode'] != 'debug':
                if self.current_round == self.global_rounds:
                    wandb.log({'ft_test_acc':test_acc})
                else:
                    wandb.log({'ft_test_acc':test_acc}, commit=False)
        except Exception:
            pass

    def test_metrics_personalized(self):
        num_samples = []
        tot_correct = []
        
        for c in self.training_clients:
            ct, ns = c.test_metrics_personalized()
            tot_correct.append(ct*1.0)

            num_samples.append(ns)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, tot_correct

    def train_metrics_personalized(self):
        num_samples = []
        losses = []
        for c in self.training_clients:
            cl, ns = c.train_metrics_personalized()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.training_clients]

        return ids, num_samples, losses