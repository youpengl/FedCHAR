
class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, dynamic_lam=0, model=None):
                    #u, g, train_data[u], test_data[u], dynamic_lam, model
        self.model = model
        self.id = id # integer
        self.group = group

        train_len = len(train_data['x'])
        # if use val dataset -> model = 'eval'
        mode = 'eval'
        if dynamic_lam or model == 'train':
            self.train_data = {'x': train_data['x'][:int(train_len * 0.9)],
                                'y': train_data['y'][:int(train_len * 0.9)]}
            self.val_data = {'x': train_data['x'][int(train_len * 0.9):],
                            'y': train_data['y'][int(train_len * 0.9):]}
        else:
            self.train_data = train_data
            self.val_data = eval_data

        self.test_data = eval_data

        self.train_samples = len(self.train_data['y'])
        self.val_samples = len(self.val_data['y'])
        self.test_samples = len(self.test_data['y'])

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()


    def get_loss(self):
        return self.model.get_loss(self.train_data)

    def get_val_loss(self):
        return self.model.get_loss(self.val_data)


    def solve_sgd(self, mini_batch_data):
        '''
        run one iteration of mini-batch SGD
        '''
        grads, loss, weights = self.model.solve_sgd(mini_batch_data)
        return (self.train_samples, weights), (self.train_samples, grads), loss

    def train_error(self):

        tot_correct, loss, _ = self.model.test(self.train_data)
        return tot_correct, loss, self.train_samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss, cm = self.model.test(self.test_data)
        return tot_correct, loss, self.test_samples, cm

    def val(self):
        '''val current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss, cm = self.model.test(self.val_data)
        return tot_correct, loss, self.test_samples, cm

    def validate(self):
        tot_correct, loss, cm = self.model.test(self.val_data)
        return tot_correct, loss, self.val_samples, cm