import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



def process_grad(grads):
    '''
    Args:
        grads: grad
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

def l2_clip(cgrads):

    # input: grads (or model updates, or models) from all selected clients
    # output: clipped grads in the same shape

    flattened_grads = []
    for i in range(len(cgrads)):
        flattened_grads.append(process_grad(cgrads[i]))
    norms = [np.linalg.norm(u) for u in flattened_grads]

    clipping_threshold = np.median(norms)

    clipped = []
    for grads in cgrads:
        norm_ = np.linalg.norm(process_grad(grads), 2)
        if norm_ > clipping_threshold:
            clipped.append([u * (clipping_threshold * 1.0) / (norm_ + 1e-10) for u in grads])
        else:
            clipped.append(grads)

    return clipped


def get_stdev(parameters):

    # input: the model parameters
    # output: the standard deviation of the flattened vector

    flattened_param = process_grad(parameters)
    return np.std(flattened_param)



