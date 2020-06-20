import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import layers
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val, precision, recall


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


model_name = 'mobilenet_extra_training_2'

if not os.path.exists("build_save_path"):
    os.makedirs("build_save_path")

if not os.path.exists("build_save_path_for_checkpoint_model"):
    os.makedirs("build_save_path_for_checkpoint_model")

if not os.path.exists("build_save_path_for_training_log_model"):
    os.makedirs("build_save_path_for_training_log_model")

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   # featurewise_center=True,
                                   # preprocessing_function=None,
                                   rotation_range=45,
                                   width_shift_range=0.2,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=[0.7, 1],
                                   brightness_range=[0.6, 1.1],
                                   channel_shift_range=60,
                                   horizontal_flip=True,
                                   # zca_whitening=True,
                                   # zca_epsilon=1e-03,
                                   # vertical_flip=True,
                                   # validation_split=0.03,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(
    rescale=1. / 255,
    # rotation_range=45,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.3,
    # zoom_range=0.1,
    # brightness_range=[0.8, 1.2],
    # channel_shift_range=10,
    # horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
    'train_folders_path',
    # target_size=(300, 300),
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    interpolation='nearest')

validation_generator = valid_datagen.flow_from_directory(
    'valid_folders_path',
    # target_size=(300, 300),
    target_size=(224, 224),
    batch_size=32,
    color_mode='rgb',
    shuffle=False,
    class_mode='categorical',
    interpolation='nearest')

# checking data augmenting
# for i in range(5):
#     s = train_generator.next()
#     for i in range(s[0].shape[0]):
#         im = s[0][i, :, :, :]
#         RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#         cv2.imshow('', RGB_img)
#         cv2.waitKey(0)

# inputs_shape = (300, 300, 3)

# mobile_1 = MobileNetV2(input_shape=(300, 300, 3), include_top=False,
#                        alpha=0.75, weights='imagenet', pooling='avg')

# mobile_2 = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
mobile_2 = MobileNetV2(input_shape=(224, 224, 3), include_top=True, weights='imagenet', classes=8)

# create model with mobile 2 conv layers
# model = Sequential([
#     mobile_2,
#     AveragePooling2D(pool_size=(7, 7)),
#     Flatten(),
#     layers.Dropout((0.1)),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout((0.1)),
#     Dense(8, activation='softmax')
# ])

class_loss_weights = {0: 3, 1: 7.98, 2: 7.79, 3: 5.82, 4: 1.11,
                      5: 1, 6: 2.94, 7: 3.54}

mobile_2.compile(loss='categorical_crossentropy',
                 # optimizer=SGD(lr=0.0005),
                 optimizer=Adam(lr=0.0001),
                 metrics=['acc', get_f1, precision, recall])

# load pretrain weights
# mobile_2.load_weights('pretrain_model_path')

checkpoint_path = 'model_checkpoint_path'
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='loss', verbose=1,
                             save_best_only=False, save_weights_only=False,
                             mode='min', period=1)

csv_callback = CSVLogger(
    "model_training_log_path"
    , append=True, separator=',')
callbacks_list = [checkpoint, csv_callback]

history = mobile_2.fit_generator(
    train_generator,
    # steps_per_epoch=10000,
    epochs=50,
    verbose=2,
    shuffle=True,
    class_weight=class_loss_weights,
    callbacks=callbacks_list,
    validation_data=validation_generator)

# save model
mobile_2.save('save_model_path')
mobile_2.save_weights('save_model_path')

# plot accuracy and loss for model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Training and Validation Accuracy')
plt.savefig("save_model_path")

plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.title('Training and Validation Loss')
plt.savefig("save_model_path")
# plt.show()


# confusion matrix
# predictions = mobile_2.predict_generator(train_generator)
# predicted_classes = np.argmax(predictions, axis=1)
# true_classes = train_generator.classes
# class_labels = list(train_generator.class_indices.keys())
# cm = confusion_matrix(true_classes, predicted_classes)
# print(cm)

# plot confusion matrix
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(cm)
# plt.title('Confusion matrix of the classifier')
# fig.colorbar(cax)
# ax.set_xticklabels([''] + class_labels)
# ax.set_yticklabels([''] + class_labels)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()

print('finish')