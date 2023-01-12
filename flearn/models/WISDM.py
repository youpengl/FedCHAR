import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.keras.layers import LSTM, Dense, Flatten, Dropout, Conv2D,  MaxPooling2D, TimeDistributed

class Model(object):
    def __init__(self, num_classes, q, optimizer, seed=1):
        # model_params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(0)
            self.features, self.labels, self.train_op, self.grads, self.loss, self.eval_metric_ops, self.cm = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
    def create_model(self, q, optimizer):
        """Model function for CNN."""
        SEGMENT_TIME_SIZE = 200
        N_FEATURES = 3
        N_HIDDEN_NEURONS = 30
        N_CLASSES = 6
        L2_LOSS = 0.0015


        features = tf.placeholder(tf.float32, shape=[None, SEGMENT_TIME_SIZE * N_FEATURES], name='features')

        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')
        W = {
            'hidden': tf.Variable(tf.random_normal([N_FEATURES, N_HIDDEN_NEURONS])),
            'output': tf.Variable(tf.random_normal([N_HIDDEN_NEURONS, N_CLASSES]))
        }

        b = {
            'hidden': tf.Variable(tf.random_normal([N_HIDDEN_NEURONS], mean=1.0)),
            'output': tf.Variable(tf.Variable(tf.random_normal([N_CLASSES])))
        }


        X = tf.reshape(features, [-1, SEGMENT_TIME_SIZE, N_FEATURES])
        X = tf.transpose(X, [1, 0, 2])
        X = tf.reshape(X, [-1, N_FEATURES])

        hidden = tf.nn.relu(tf.matmul(X, W['hidden']) + b['hidden'])
        hidden = tf.split(hidden, SEGMENT_TIME_SIZE, 0)

        # Stack two LSTM cells on top of each other
        lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_NEURONS, forget_bias=1.0)
        lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_NEURONS, forget_bias=1.0)
        lstm_layers = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_1, lstm_cell_2])

        outputs, _ = tf.nn.static_rnn(lstm_layers, hidden, dtype=tf.float32)

        # Get output for the last time step from a "many to one" architecture
        last_output = outputs[-1]

        logits = tf.matmul(last_output, W['output'] + b['output'])

        predictions = {
          "classes": tf.argmax(input=logits, axis=1)
        }

        l2 = L2_LOSS * sum(tf.nn.l2_loss(i) for i in tf.trainable_variables())
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + l2
        grads_and_vars = optimizer.compute_gradients(loss)
        grads, _ = zip(*grads_and_vars)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        cm = tf.confusion_matrix(labels, predictions["classes"], self.num_classes)
        return features, labels, train_op, grads, loss, eval_metric_ops, cm

    def set_params(self, model_params=None):
        if model_params is not None:
            with self.graph.as_default():
                all_vars = tf.trainable_variables()
                for variable, value in zip(all_vars, model_params):
                    variable.load(value, self.sess)

    def get_params(self):
        with self.graph.as_default():
            model_params = self.sess.run(tf.trainable_variables())
        return model_params

    def solve_sgd(self, mini_batch_data):
        with self.graph.as_default():
            grads, loss, _ = self.sess.run([self.grads, self.loss, self.train_op],
                                           feed_dict={self.features: mini_batch_data[0],
                                                      self.labels: mini_batch_data[1]})

        weights = self.get_params()
        return grads, loss, weights

    def get_loss(self, data):
        with self.graph.as_default():
            loss = self.sess.run(self.loss, feed_dict={self.features: data['x'], self.labels: data['y']})
        return loss

    def test(self, data):
        '''
        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        '''
        with self.graph.as_default():
            tot_correct, loss, cm = self.sess.run([self.eval_metric_ops, self.loss, self.cm],
                                              feed_dict={self.features: data['x'], self.labels: data['y']})
        return tot_correct, loss, cm

    def close(self):
        self.sess.close()