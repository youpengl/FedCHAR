import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model(object):
    def __init__(self, num_classes, q, optimizer, seed=1):
        # model_params
        self.num_classes = num_classes

        # create computation graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(seed)
            self.features, self.labels, self.train_op, self.grads, self.loss, self.eval_metric_ops, self.cm = self.create_model(q, optimizer)
            self.saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())

    def create_model(self, q, optimizer):
        """Model function for DNN."""
        features = tf.placeholder(tf.float32, shape=[None, 900], name='features')
        labels = tf.placeholder(tf.int64, shape=[None, ], name='labels')

        # dense = tf.layers.dense(inputs=features, units=300, activation=tf.nn.relu)
        # logits = tf.layers.dense(inputs=dense, units=self.num_classes)

        dense = tf.layers.dense(inputs=features, units=300, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=self.num_classes)
        predictions = {"classes": tf.argmax(input=logits, axis=1)}
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
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