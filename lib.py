from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from tensorflow.keras import backend as K
import traceback
import tensorflow.compat.v1 as tf_1
import cv2
import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import efficientnet.keras as e
import itertools


def plot_model(history, save_dir):
    # plot accuracy and loss for model
    r2_score = history.history['r2']
    val_r2_score = history.history['val_r2']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    # plt.subplot(2, 1, 1)
    plt.plot(r2_score, label='Training r2_score')
    plt.plot(val_r2_score, label='Validation r2_score')
    plt.legend(loc='upper right')
    plt.ylabel('R2_score')
    plt.xlabel('Epoch')
    plt.title('Training and Validation R2_score')
    # plt.savefig(f"./models/{model_name}/r2_score_fig.jpg")
    plt.savefig(save_dir + "r2_score_fig.jpg")
    plt.close()

    # plt.subplot(2, 1, 2)
    plt.figure(figsize=(8, 8))
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')
    # plt.savefig(f"./models/{model_name}/Loss_fig.jpg")
    plt.savefig(save_dir + "Loss_fig.jpg")
    plt.close()


def strided_axis0(a, l):
    # INPUTS :
    # a is array
    # l is length of array along axis=0 to be cut for forming each subarray

    # Length of 3D output array along its axis=0
    nd0 = a.shape[0] - l + 1

    # Store shape and strides info
    # m0, m1, m2, m3 = a.shape
    m = list(a.shape)
    # s0, s1, s2, s3 = a.strides
    s = list(a.strides)

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(a, shape=tuple([nd0, l] + m[1:]), strides=tuple([s[0]] + s))


def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def f1_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_macro_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


def precision(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def r2(y_true, y_pred):
    ss_res = K.sum(K.square(y_true - y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - ss_res / (ss_tot + K.epsilon())


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def ccc(y_true, y_pred):
    x = y_true
    y = y_pred

    x_mean = K.mean(x)
    y_mean = K.mean(y)

    cov = K.mean(tf.multiply((x - x_mean), (y - y_mean)))
    r_num = 2 * cov
    r_den = K.mean(K.square(x - x_mean)) + K.mean(K.square(y - y_mean)) + K.square(x_mean - y_mean)

    ccc = r_num / (r_den + K.epsilon())
    return ccc


def ccc_loss(y_true, y_pred):
    x = y_true
    y = y_pred

    x_mean = K.mean(x)
    y_mean = K.mean(y)

    cov = K.mean(tf.multiply((x - x_mean), (y - y_mean)))
    r_num = 2 * cov
    r_den = K.mean(K.square(x - x_mean)) + K.mean(K.square(y - y_mean)) + K.square(x_mean - y_mean)

    ccc = r_num / (r_den + K.epsilon())

    return K.minimum(1 - ccc, 1.0)


def tukey_biweight_loss(y_true, y_pred, c=4.685):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < c
    quadratic_loss = ((c ** 2) / 6) * (1 - K.pow((1 - K.pow((error / c), 2)), 3))
    linear_loss = c ** 2 / 6
    return tf.keras.backend.mean(tf.where(cond, quadratic_loss, linear_loss))


def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def weightedLoss(true, pred, weightsList):
    axis = -1  # if channels last
    # axis=  1 #if channels first

    # argmax returns the index of the element with the greatest value
    # done in the class axis, it returns the class index
    classSelectors = K.argmax(true, axis=axis)
    # if your loss is sparse, use only true as classSelectors

    # considering weights are ordered by class, for each class
    # true(1) if the class index is equal to the weight index
    classSelectors = [K.equal(tf.cast(i, tf.int64), tf.cast(classSelectors, tf.int64)) for i in range(len(weightsList))]

    # casting boolean to float for calculations
    # each tensor in the list contains 1 where ground true class is equal to its index
    # if you sum all these, you will get a tensor full of ones.
    classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

    # for each of the selections above, multiply their respective weight
    weights = [sel * w for sel, w in zip(classSelectors, weightsList)]

    # sums all the selections
    # result is a tensor with the respective weight for each element in predictions
    weightMultiplier = weights[0]
    for i in range(1, len(weights)):
        weightMultiplier = weightMultiplier + weights[i]

    # make sure your originalLossFunc only collapses the class axis
    # you need the other axes intact to multiply the weights tensor
    loss = K.categorical_crossentropy(true, pred)
    loss = loss * weightMultiplier
    return loss


def convert_bb_shape(bb):
    """

    :param bb: 2-d array format (x1, y1, x2, y2)
    :return: 2-d array format (x, y, w, h)
    """

    bb[:, 2] = np.abs(bb[:, 0] - bb[:, 2])
    bb[:, 3] = np.abs(bb[:, 1] - bb[:, 3])

    return bb


def prepare_image(image, bb, shape, convert_shape=False):
    """
    :param image:
    :param bb: 1-d array: [x1, y1, x2, y2] or [x, y, w, h] based on convert_shape
    :return: extracted square shape x-box with respect to bb, size of x-box before resizing to shape size
    """
    check_bb = bb.copy()
    if convert_shape:
        bb = convert_bb_shape(bb.reshape((1, 4)))

    bb = bb.flatten()

    # side of the square
    # s = int(max(bb[2], bb[3]))
    # s = int(min(bb[2], bb[3]))
    # s = int((bb[2] + bb[3])/2)
    s = int(max(bb[2], bb[3]) * 0.58)
    # s = int(max(bb[2], bb[3])*np.random.uniform(low=0.6, high=1))

    height = image.shape[0]
    width = image.shape[1]

    # center of bb : (Cx, Cy)
    center = [bb[0] + bb[2] / 2, bb[1] + bb[3] / 2]

    center[0] = int(center[0])
    center[1] = int(center[1])

    # plot_circle(image, center)

    # check side of the square is not bigger than frame height and frame width
    if 2 * s > width or 2 * s > height:
        s = int(min(width, height) / 2)

    if (center[0] - s) < 0:
        center[0] = center[0] + abs(center[0] - s)
        # plot_circle(image, center)

    if (center[0] + s) > width:
        center[0] = center[0] - (center[0] + s - width)
        # plot_circle(image, center)

    if (center[1] - s) < 0:
        center[1] = center[1] + abs(center[1] - s)
        # plot_circle(image, center)

    if (center[1] + s) > height:
        center[1] = center[1] - (center[1] + s - height)
        # plot_circle(image, center)

    new_image = image[center[1] - s:center[1] + s, center[0] - s:center[0] + s]

    h = new_image.shape[0]
    w = new_image.shape[1]

    # cv2.imshow('new image', cv2.resize(image, (900, 900), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('new image', image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # start = time.time()
    try:
        new_image = cv2.resize(new_image, shape, interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(check_bb)
        cv2.imshow('new image', new_image)
        cv2.waitKey()
        traceback.print_exc()

    # end = time.time()
    # print(f'resizing image takes {(end-start)*1000:.2f} milli sec')

    return new_image, [center[0] - s, center[1] - s, w, h]


class FaceDetector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf_1.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf_1.GraphDef()
            graph_def.ParseFromString(f.read())

        graph = tf_1.Graph()
        with graph.as_default():
            tf_1.import_graph_def(graph_def, name='import')

        self.input_image = graph.get_tensor_by_name('import/image_tensor:0')
        self.output_ops = [
            graph.get_tensor_by_name('import/boxes:0'),
            graph.get_tensor_by_name('import/scores:0'),
            graph.get_tensor_by_name('import/num_boxes:0'),
        ]

        gpu_options = tf_1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf_1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf_1.Session(graph=graph, config=config_proto)

    def __call__(self, image, score_threshold=0.5):
        """Detect faces.
        Arguments:
            image: a numpy uint8 array with shape [height, width, 3],
                that represents a RGB image.
            score_threshold: a float number.
        Returns:
            boxes: a float numpy array of shape [num_faces, 4].
            scores: a float numpy array of shape [num_faces].
        Note that box coordinates are in the order: ymin, xmin, ymax, xmax!
        """
        if len(image.shape) == 3:
            h, w, _ = image.shape
            image = np.expand_dims(image, 0)
        elif len(image.shape) == 4:
            _, h, w, _ = image.shape
        else:
            print('input shape is wrong')
        boxes, scores, num_boxes = self.sess.run(self.output_ops, feed_dict={self.input_image: image})
        # num_boxes = num_boxes[0]
        # boxes = boxes[0][:num_boxes]
        # scores = scores[0][:num_boxes]
        scores = np.ma.filled(np.ma.masked_where(scores < score_threshold, scores), 0)

        # to_keep = scores > score_threshold

        # boxes = boxes[to_keep]
        # scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        return boxes, scores


def load(path):
    """
    takes as input the path to a .pts and returns a list of
    tuples of floats containing the points in in the form:
    [(x_0, y_0, z_0),
    (x_1, y_1, z_1),
    ...
    (x_n, y_n, z_n)]
    """
    with open(path) as f:
        rows = [rows.strip() for rows in f]

    """Use the curly braces to find the start and end of the point data"""
    head = rows.index('{') + 1
    tail = rows.index('}')

    """Select the point data split into coordinates"""
    raw_points = rows[head:tail]
    coords_set = [point.split() for point in raw_points]

    """Convert entries from lists of strings to tuples of floats"""
    points = [tuple([float(point) for point in coords]) for coords in coords_set]
    return points


def attention_3d_block(hidden_states, dense_activation='tanh'):
    """
    Many-to-one attention mechanism for Keras.
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, 128)
    @author: felixhao28.
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation=dense_activation, name='attention_vector')(
        pre_activation)
    return attention_vector
