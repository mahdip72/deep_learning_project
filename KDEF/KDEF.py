from tensorflow import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import Sequential, Model
from tensorflow import convert_to_tensor
from STN import SpatialTransformer
# import efficientnet.tfkeras as e
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import imgaug.augmenters as iaa


class RandAug:
    def __init__(self,
                 number_of_affine,
                 magnitude_of_affine):
        self.aug = iaa.RandAugment(n=number_of_affine, m=magnitude_of_affine)

    def transform(self, input_im):
        input_im = input_im.astype(np.uint8)
        aug_im = self.aug.augment(image=input_im)
        return aug_im.astype(np.float64)


def custom_augment(input_im):
    gate = 4
    # gate = random.randrange(1, 8, 1)
    if gate == 1 or gate == 2:
        k = random.randrange(1, 6, 2)
        aug_im = cv2.GaussianBlur(input_im, (k, k), 0)
        return aug_im
    elif gate == 3:
        row, col, ch = input_im.shape
        mean = 0
        var = 0.09
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss * 255
        aug_im = np.add(input_im, gauss)
        return aug_im
    elif gate == 4 or gate == 5:
        input_im = input_im / 255
        vals = len(np.unique(input_im))
        vals = 2 ** np.ceil(np.log2(vals))
        aug_im = np.random.poisson(input_im * vals) / float(vals)
        return aug_im * 255
    elif gate == 6:
        # input_tensor = convert_to_tensor(input_im)
        # aug_im = image.random_jpeg_quality(input_im, 50, 100)
        # aug_im = image.decode_image(aug_im)

        aug_im = image.adjust_brightness(input_im, 100)
        # aug_im = image.rgb_to_grayscale(input_im, name=None)
        # aug_im = image.random_contrast(input_im, 1, 5)
        # aug_im = image.random_brightness(input_im, 50)
        # aug_im = image.random_hue(input_im, 0.05)
        # aug_im = image.per_image_standardization(input_im)
        # aug_im = image.draw_bounding_boxes(input_im[np.newaxis, :, :, :],
        #                                    np.array([[[0.1, 0.1, 0.5, 0.5]]]),
        #                                    colors=np.zeros([224, 224]))
        return aug_im
    else:
        return input_im


model_name = 'noisy_efficientb0'

if not os.path.exists(f"./models/{model_name}"):
    os.makedirs(f"./models/{model_name}")

# aug = RandAug(number_of_affine=2, magnitude_of_affine=(0, 10))

train_datagen = ImageDataGenerator(rescale=1./255,
                                   # preprocessing_function=aug.transform,
                                   preprocessing_function=custom_augment,
                                   # rotation_range=45,
                                   # width_shift_range=0.1,
                                   # height_shift_range=0.1,
                                   # shear_range=0.1,
                                   # zoom_range=[0.8, 1],
                                   # brightness_range=[0.8, 1.2],
                                   # channel_shift_range=30,
                                   # horizontal_flip=True,
                                   # vertical_flip=True,
                                   validation_split=0.15,
                                   fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
    # '/gdrive/My Drive/Colab Notebooks/FER/KDEF/Dataset',
    # '/mnt/storage/mehdi_dataset/KDEF/',
    'E:\Document\PyCharm Project\FER\KDEF\KDEF',
    target_size=(220, 220),
    # target_size=(220, 300),
    batch_size=16,
    color_mode='rgb',
    shuffle=True,
    class_mode='categorical',
    subset='training',
    interpolation='nearest')

validation_generator = train_datagen.flow_from_directory(
        # '/mnt/storage/mehdi_dataset/KDEF/',
        'E:\Document\PyCharm Project\FER\KDEF\KDEF',
        target_size=(220, 220),
        batch_size=16,
        color_mode='rgb',
        # shuffle=True,
        class_mode='categorical',
        subset='validation',
        interpolation='nearest')

for i in range(5):
    s = train_generator.next()
    for j in range(s[0].shape[0]):
        im = s[0][j, :, :, :]
        RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cv2.imshow('ss', RGB_img)
        cv2.waitKey(0)

# inputs_shape = (562, 762, 3)
inputs_shape = (220, 220, 3)

# initial weights
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 1
b[1, 1] = 1
W = np.zeros((64, 6), dtype='float32')
weights = [W, b.flatten()]

mobile = MobileNetV2(input_shape=inputs_shape, alpha=0.5, include_top=False, weights=None, pooling='max')
model = MobileNetV2(input_shape=inputs_shape, alpha=0.5, include_top=True, classes=7, weights=None, pooling='max')
# nasnet = NASNetMobile(input_shape=inputs_shape, include_top=False, weights=None)
# model = e.EfficientNetB0(include_top=False,
#                          input_shape=(128, 128, 3),
#                          weights=None,
#                          pooling='max')
                         # classes=7)

locnet = Sequential([
    mobile,
    # layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(6, weights=weights)
])

locnet.summary()

FER = Sequential()
FER.add(SpatialTransformer(localization_net=locnet,
                           output_size=(128, 128),
                           input_shape=inputs_shape))
FER.add(model)
# model.add(AveragePooling2D(pool_size=(2, 2)))
# FER.add(layers.Flatten())

# FER.add(layers.Dense(128, activation='relu'))
# FER.add(layers.Dense(7, activation='softmax'))

FER.summary()
FER.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc'])

# load pretrain weights
# model.load_weights('/mnt/storage/mehdi_dataset/KDEF/models/model3_weights.h5')
# by_name=True


checkpoint = ModelCheckpoint(f'./models/{model_name}/check_model.h5', monitor='val_acc', verbose=1,
                             save_best_only=True, save_weights_only=False, mode='max', period=1)
csv_callback = CSVLogger(f'./models/{model_name}/training_log.csv', append=True, separator=',')
callbacks = [checkpoint, csv_callback]

history = FER.fit_generator(
      train_generator,
      steps_per_epoch=120,
      epochs=20,
      verbose=1,
      shuffle=True,
      # callbacks=callbacks,
      validation_data=validation_generator)
      # validation_steps=21)


# save model
model.save(f'./models/{model_name}/final.h5')

# plot accuracy and loss for model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.savefig(f'./models/{model_name}/fig.png')
plt.show()
