import tensorflow as tf
import numpy as np
import os
import random
from sklearn.model_selection import KFold
from lib import attention_3d_block


class Generator:
    def __init__(self, data_list, path):
        self.data_list = data_list
        self.path = path

    def flow(self):
        random.shuffle(self.data_list)
        # while True:
        for subject in self.data_list:
            sample_list = os.listdir(os.path.join(self.path, subject))
            for sample in sample_list:
                data = np.load(os.path.join(self.path, subject, sample))
                x = data['arr_0']
                y = data['arr_1'] - 1
                yield x[np.newaxis, :, :], y[np.newaxis, :]


tf.autograph.set_verbosity(0)
auto = tf.data.experimental.AUTOTUNE

dataset_path = 'dataset/'
subject_list = os.listdir(dataset_path)

kf = KFold(n_splits=10, shuffle=True)
kf.get_n_splits(subject_list)

best_acc = []
fold = 0
for train_index, valid_index in kf.split(subject_list):
    fold += 1
    train_subject = list(np.asarray(subject_list)[train_index])
    valid_subject = list(np.asarray(subject_list)[valid_index])

    train_gen = Generator(train_subject, path=dataset_path)
    train_ds = tf.data.Dataset.from_generator(train_gen.flow, (tf.float16, tf.float16),
                                              (tf.TensorShape([None, None, 1280]),
                                               tf.TensorShape([None, 1]))).prefetch(auto)

    valid_gen = Generator(valid_subject, path=dataset_path)
    valid_ds = tf.data.Dataset.from_generator(valid_gen.flow, (tf.float16, tf.float16),
                                              (tf.TensorShape([None, None, 1280]),
                                               tf.TensorShape([None, 1]))).prefetch(auto)

    # for i, j in train_gen.flow():
    #     print(j)

    # for i, j in train_ds:
    #     print(i.shape)
    #     print(j.shape)

    tf.keras.backend.clear_session()
    input_model = tf.keras.layers.Input(shape=(None, 1280))
    drop_out = tf.keras.layers.AlphaDropout(0.5)
    # x_1 = tf.keras.layers.GRU(64, activation='selu', return_sequences=True)(input_model)
    x_1 = attention_3d_block(input_model, dense_activation='selu')
    x_2 = tf.keras.layers.Dense(16, activation='selu')(x_1)
    x_3 = tf.keras.layers.Dense(7, activation='softmax')(x_1)

    model = tf.keras.Model(input_model, x_3)
    # model.summary()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,
                                                                 decay_steps=400,
                                                                 decay_rate=0.8)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    # class_loss_weights = {0: 3, 1: 7.98, 2: 7.79, 3: 5.82, 4: 1.11,
    #                       5: 1, 6: 2.94}

    history = model.fit(train_ds,
                        epochs=15,
                        # steps_per_epoch=int(len(train_path)),
                        validation_data=valid_ds,
                        # class_weight=class_loss_weights,
                        # validation_steps=int(len(valid_path)),
                        verbose=0)

    best_acc.append(max(history.history['val_acc']))
    print(f'validation acc in iteration {fold}:', max(history.history['val_acc']))
    del model

print('10-fold validation acc:', np.mean(best_acc))
with open('result.txt', 'a') as file:
    # file.write('description, k-fold (mean acc)')
    file.write('\n')
    file.write(f'"attention_selu_sgd_shuffle", {np.mean(best_acc)}')

print('finish')
