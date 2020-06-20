import pickle
import tensorflow as tf
import numpy as np
import efficientnet.tfkeras as e
import os
from collections import Counter
import collections, functools, operator
import random
import cv2
from skimage.util import random_noise
from sklearn.model_selection import KFold
from tensorflow import keras as k
from utils import f1_macro, f1_macro_loss


def calculate_class_weights(data):
    class_weights = dict()
    keys = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26']

    first = True
    for dic in data:
        if first:
            class_weights = dic.copy()
            first = False
            continue
        else:
            for key in class_weights:
                class_weights[key] = dict(functools.reduce(operator.add, map(collections.Counter,
                                                                             [dic[key], class_weights[key]])))
    num_samples = sum(class_weights['AU1'].values())
    for key in keys:
        dic = class_weights[key].copy()
        maximum = max(dic.values())
        for j in dic:
            if dic[j] != 0:
                class_weights[key][j] = maximum / dic[j]
            else:
                class_weights[key][j] = 1

    # for key in keys:
    #     au = {}
    #     for i, j in enumerate(vars()[key]):
    #         au[i] = j
    #     class_weights[key] = au

    return class_weights, num_samples


def augment(image_1):
    image_2 = image_1.numpy()
    image_2 = tf.keras.preprocessing.image.random_rotation(image_2, 15, fill_mode='nearest',
                                                           row_axis=0, col_axis=1, channel_axis=2)
    image_2 = tf.keras.preprocessing.image.random_zoom(image_2, [0.9, 1], fill_mode='nearest',
                                                       row_axis=0, col_axis=1, channel_axis=2)
    image_2 = tf.keras.preprocessing.image.random_shear(image_2, 5, fill_mode='nearest',
                                                        row_axis=0, col_axis=1, channel_axis=2)
    gate_2 = random.randrange(1, 8, 1)
    # gate_2 = 1
    if gate_2 == 1:
        image_2 = random_noise(image_2, mode="gaussian")
    elif gate_2 == 2:
        image_2 = random_noise(image_2, mode="poisson")
    elif gate_2 == 3:
        image_2 = cv2.resize(cv2.resize(image_2, (128, 128)), (224, 224))
    elif gate_2 == 4:
        rand = random.randrange(1, 8, 2)
        image_2 = cv2.GaussianBlur(image_2, (rand, rand), 0)
    return image_2


def image_augmenting_train(image, label):
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [224, 224])
    # image = tf.py_function(func=augment, inp=[image], Tout=tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    return image, label


def image_augmenting_valid(image, label):
    # image = tf.image.random_hue(image, 0.05)
    # image = tf.image.random_flip_left_right(image)
    # image = tf.image.resize(image, [224, 224])
    # image = tf.py_function(func=augment, inp=[image], Tout=tf.float32)
    image = tf.reshape(image, [224, 224, 3])
    return image, label


def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'AU1': tf.io.FixedLenFeature((), tf.string),
        'AU2': tf.io.FixedLenFeature((), tf.string),
        'AU4': tf.io.FixedLenFeature((), tf.string),
        'AU5': tf.io.FixedLenFeature((), tf.string),
        'AU6': tf.io.FixedLenFeature((), tf.string),
        'AU9': tf.io.FixedLenFeature((), tf.string),
        'AU12': tf.io.FixedLenFeature((), tf.string),
        'AU15': tf.io.FixedLenFeature((), tf.string),
        'AU17': tf.io.FixedLenFeature((), tf.string),
        'AU20': tf.io.FixedLenFeature((), tf.string),
        'AU25': tf.io.FixedLenFeature((), tf.string),
        'AU26': tf.io.FixedLenFeature((), tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=float)

    return image, (tf.reshape(tf.io.parse_tensor(example['AU1'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU2'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU4'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU5'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU6'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU9'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU12'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU15'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU17'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU20'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU25'], out_type=tf.float32), [6]),
                   tf.reshape(tf.io.parse_tensor(example['AU26'], out_type=tf.float32), [6])),


model_name = 'model_name'
save_path = os.path.abspath('save_path')
model_path = os.path.join(save_path, f"{model_name}")

if not os.path.exists(model_path):
    os.makedirs(model_path)

auto = tf.data.experimental.AUTOTUNE
batch = 32

tfrecord_data_path = os.path.abspath('tfrecords_data_path')
lst = os.listdir(tfrecord_data_path)

directories = []
for i in lst:
    if os.path.isdir(os.path.join(tfrecord_data_path, i)):
        directories.append(i)

kf = KFold(n_splits=9, shuffle=False)
kf.get_n_splits(directories)

val_loss = []
val_accs = []
fold = 0
for train_index, valid_index in kf.split(directories):
    fold += 1
    train_subject = list(np.asarray(directories)[train_index])
    valid_subject = list(np.asarray(directories)[valid_index])

    train_subject_path = [os.path.abspath(os.path.join(tfrecord_data_path, subject, subject + '.tfrecords')) for subject
                          in train_subject]
    valid_subject_path = [os.path.abspath(os.path.join(tfrecord_data_path, subject, subject + '.tfrecords')) for subject
                          in valid_subject]

    train_ds = tf.data.TFRecordDataset(train_subject_path)
    train_ds = train_ds.map(read_tfrecord).map(image_augmenting_train).batch(batch).prefetch(auto).repeat()
    # train_ds = train_ds.map(read_tfrecord).batch(batch).prefetch(auto)

    valid_ds = tf.data.TFRecordDataset(valid_subject_path)
    valid_ds = valid_ds.map(read_tfrecord).map(image_augmenting_valid).batch(batch).prefetch(auto).repeat()

    # testing tfrecord data
    # for i in train_ds.take(1):
    #     print(i[0].shape)
    #     im = (i[0].numpy()*255).astype(np.uint8)
    #     cv2.imshow('', im)
    #     cv2.waitKey(0)

    train_class_num_path = [os.path.abspath(os.path.join(tfrecord_data_path, subject, subject + '.pkl')) for subject in
                            train_subject]
    valid_class_num_path = [os.path.abspath(os.path.join(tfrecord_data_path, subject, subject + '.pkl')) for subject in
                            valid_subject]

    train_class = []
    for path in train_class_num_path:
        with open(path, "rb") as a_file:
            train_class.append(pickle.load(a_file))

    class_weights, num_samples_train = calculate_class_weights(train_class)

    valid_class = []
    for path in valid_class_num_path:
        with open(path, "rb") as a_file:
            valid_class.append(pickle.load(a_file))

    _, num_samples_valid = calculate_class_weights(valid_class)

    fer_model_path = 'pretrain model path'
    model = k.models.load_model(fer_model_path, compile=False)
    # layer_name = 'global_average_pooling2d'
    layer_name = 'efficientnet-b0'  # last conv layer
    conv_model = k.models.Model(inputs=model.get_layer(layer_name).input,
                                outputs=model.get_layer(layer_name).output)
    # conv_model.summary()

    tf.keras.backend.clear_session()
    x1 = tf.keras.layers.Input(shape=(224, 224, 3))
    x2 = conv_model(x1)
    x3 = tf.keras.layers.GlobalAveragePooling2D()(x2)
    x3 = tf.keras.layers.Dropout(0.4)(x3)
    # x3 = tf.keras.layers.Dense(128, activation='selu')(x2)
    out_1 = tf.keras.layers.Dense(6, activation='softmax', name='AU1')(x3)
    out_2 = tf.keras.layers.Dense(6, activation='softmax', name='AU2')(x3)
    out_3 = tf.keras.layers.Dense(6, activation='softmax', name='AU4')(x3)
    out_4 = tf.keras.layers.Dense(6, activation='softmax', name='AU5')(x3)
    out_5 = tf.keras.layers.Dense(6, activation='softmax', name='AU6')(x3)
    out_6 = tf.keras.layers.Dense(6, activation='softmax', name='AU9')(x3)
    out_7 = tf.keras.layers.Dense(6, activation='softmax', name='AU12')(x3)
    out_8 = tf.keras.layers.Dense(6, activation='softmax', name='AU15')(x3)
    out_9 = tf.keras.layers.Dense(6, activation='softmax', name='AU17')(x3)
    out_10 = tf.keras.layers.Dense(6, activation='softmax', name='AU20')(x3)
    out_11 = tf.keras.layers.Dense(6, activation='softmax', name='AU25')(x3)
    out_12 = tf.keras.layers.Dense(6, activation='softmax', name='AU26')(x3)

    outs = [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12]

    au_model = tf.keras.models.Model(x1, outs)
    # au_model.summary()

    losses = {'AU1': 'categorical_crossentropy',
              'AU2': 'categorical_crossentropy',
              'AU4': 'categorical_crossentropy',
              'AU5': 'categorical_crossentropy',
              'AU6': 'categorical_crossentropy',
              'AU9': 'categorical_crossentropy',
              'AU12': 'categorical_crossentropy',
              'AU15': 'categorical_crossentropy',
              'AU17': 'categorical_crossentropy',
              'AU20': 'categorical_crossentropy',
              'AU25': 'categorical_crossentropy',
              'AU26': 'categorical_crossentropy'}

    # create custom loss
    # loss1 = partial(f1_macro_loss)
    # loss2 = partial(f1_macro_loss)
    # loss4 = partial(f1_macro_loss)
    # loss5 = partial(f1_macro_loss)
    # loss6 = partial(f1_macro_loss)
    # loss9 = partial(f1_macro_loss)
    # loss12 = partial(f1_macro_loss)
    # loss15 = partial(f1_macro_loss)
    # loss17 = partial(f1_macro_loss)
    # loss20 = partial(f1_macro_loss)
    # loss25 = partial(f1_macro_loss)
    # loss26 = partial(f1_macro_loss)

    # loss1.__name__ = 'loss1'
    # loss2.__name__ = 'loss2'
    # loss4.__name__ = 'loss4'
    # loss5.__name__ = 'loss5'
    # loss6.__name__ = 'loss6'
    # loss9.__name__ = 'loss9'
    # loss12.__name__ = 'loss12'
    # loss15.__name__ = 'loss15'
    # loss17.__name__ = 'loss17'
    # loss20.__name__ = 'loss20'
    # loss25.__name__ = 'loss25'
    # loss26.__name__ = 'loss26'

    # losses = {'AU1': loss1,
    #           'AU2': loss2,
    #           'AU4': loss4,
    #           'AU5': loss5,
    #           'AU6': loss6,
    #           'AU9': loss9,
    #           'AU12': loss12,
    #           'AU15': loss15,
    #           'AU17': loss17,
    #           'AU20': loss20,
    #           'AU25': loss25,
    #           'AU26': loss26}

    metrics = {'AU1': ['acc', f1_macro],
               'AU2': ['acc', f1_macro],
               'AU4': ['acc', f1_macro],
               'AU5': ['acc', f1_macro],
               'AU6': ['acc', f1_macro],
               'AU9': ['acc', f1_macro],
               'AU12': ['acc', f1_macro],
               'AU15': ['acc', f1_macro],
               'AU17': ['acc', f1_macro],
               'AU20': ['acc', f1_macro],
               'AU25': ['acc', f1_macro],
               'AU26': ['acc', f1_macro]}

    loss_weight = {'AU1': 0.083,
                   'AU2': 0.083,
                   'AU4': 0.083,
                   'AU5': 0.083,
                   'AU6': 0.083,
                   'AU9': 0.083,
                   'AU12': 0.083,
                   'AU15': 0.083,
                   'AU17': 0.083,
                   'AU20': 0.083,
                   'AU25': 0.083,
                   'AU26': 0.083}

    au_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                     metrics=metrics,
                     loss_weights=loss_weight,
                     loss=losses)

    checkpoint_path = os.path.join(model_path, f'best_val_loss_fold_{fold}.h5')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_loss', verbose=0,
                                                    save_best_only=True, save_weights_only=False,
                                                    mode='min', save_freq='epoch')

    csv_callback = tf.keras.callbacks.CSVLogger(os.path.join(model_path, f"training_log{fold}.csv"),
                                                append=False)
    callbacks_list = [checkpoint, csv_callback]

    history = au_model.fit(train_ds,
                           epochs=16,
                           verbose=0,
                           steps_per_epoch=np.ceil(num_samples_train / batch),
                           use_multiprocessing=True,
                           validation_data=valid_ds,
                           validation_steps=np.ceil(num_samples_valid / batch),
                           callbacks=callbacks_list,
                           class_weight=class_weights)

    val_loss.append(min(history.history['val_loss']))
    print(f'validation acc in iteration {fold}:', min(history.history['val_loss']))

    print(f'end of fold {fold}')
    del au_model

print(f'{fold}-fold validation loss:', np.mean(val_loss))

# saving kfold results in txt file
with open('result.txt', 'a') as file:
    file.write('description, k-fold (mean val loss)')
    file.write('\n')
    file.write(f'{model_name}, {np.mean(val_loss)}')

print('finish')
