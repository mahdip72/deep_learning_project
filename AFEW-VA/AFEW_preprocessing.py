import cv2
import tensorflow as tf
import numpy as np
import os
import json
import tensorflow.keras as k
from utils import FaceDetector
from utils import prepare_image
import efficientnet.tfkeras as e
from tensorflow.keras import backend as K
from utils import r2


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


np.random.seed(10)
tf.keras.backend.clear_session()
# MODEL_PATH = '../demo/model/Face_Detection/model.pb'
# face_detector = FaceDetector(MODEL_PATH)

fer_model_path = '../demo/model/AffectNet/checkpoint.h5'
fer_model = k.models.load_model(fer_model_path, compile=False)
fer_model.compile(loss=root_mean_squared_error,
                  optimizer=tf.keras.optimizers.Adam(lr=0.0001))
fer_model.summary()

# model = k.models.Sequential()
# for layer in fer_model.layers[:-1]:
#     model.add(layer)

# fer_model.summary()
# layer_name = 'dense'
layer_name = 'global_average_pooling2d'
intermediate_layer_model = k.models.Model(inputs=fer_model.input,
                                          outputs=fer_model.get_layer(layer_name).output)
intermediate_layer_model.summary()

dataset_dir = 'E:\Document\Database\FER\AFEW-VA\dataset'
feature_dim = 1280
max_step = 0
batch = []
sample = 0

folders_dir = os.listdir(dataset_dir)
for folder in folders_dir:
    subjects_dir = os.listdir(os.path.join(dataset_dir, folder))
    try:
        subjects_dir.remove('README.md')
    except:
        pass

    for subject in subjects_dir:
        print(subject)
        sample += 1
        frames = os.listdir(os.path.join(dataset_dir, folder, subject))
        try:
            frames.remove(subject + '.json')
        except:
            pass

        json_dir = os.path.join(dataset_dir, folder, subject, subject + '.json')
        with open(json_dir) as j:
            subject_info = json.load(j)

        cv2.destroyAllWindows()
        counter = 0

        if len(frames) >= max_step:
            max_step = len(frames)

        input_images = np.zeros((0, 224, 224, 3))
        targets = np.zeros((0, 2))

        for frame in frames:
            counter += 1
            frame_dir = os.path.join(dataset_dir, folder, subject, frame)
            im = cv2.imread(frame_dir)

            landmarks = subject_info['frames'][f'{frame[:-4]}']['landmarks']
            arousal = subject_info['frames'][f'{frame[:-4]}']['arousal']
            valence = subject_info['frames'][f'{frame[:-4]}']['valence']
            maximum = np.amax(landmarks, axis=0)
            minimum = np.amin(landmarks, axis=0)
            bb = np.concatenate([minimum, maximum], 0)
            try:
                new_im, _ = prepare_image(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), bb, shape=(224, 224), convert_shape=True)
            except:
                print('error')
                print(subject)
                print(frame)
            input_images = np.vstack((input_images, new_im[np.newaxis, :, :, :]))

            targets = np.vstack((targets, np.array([[valence, arousal]])))

        # for i in range(input_images.shape[0]):
        #     pic = input_images[i, :, :, :].astype(np.uint8)
        #     cv2.imshow(f'frame', pic)
        #     cv2.waitKey(50)
        input_images = input_images / 255
        gru_input = intermediate_layer_model.predict(input_images)
        batch.append([gru_input, targets])
        # gru_input = strided_axis0(gru_input, max_time_steps)
        # targets = strided_axis0(targets/10, max_time_steps)
        # if len(batch) == 50 or sample == 350:
        if len(batch) == 600:
            batch_size = 0
            targets_batch = np.zeros((0, max_step, 2))
            gru_input_batch = np.zeros((0, max_step, feature_dim))
            for i in range(len(batch)):
                data = batch[i]
                subtract = max_step - data[0].shape[0]
                if subtract != 0:
                    gru_input_temp = np.zeros((max_step, data[0].shape[1]))
                    gru_input_temp[:-subtract, :] = data[0]
                    targets_temp = np.zeros((max_step, data[1].shape[1]))
                    targets_temp[:-subtract, :] = data[1]
                else:
                    gru_input_temp = data[0]
                    targets_temp = data[1]

                gru_input_batch = np.vstack((gru_input_batch, gru_input_temp[np.newaxis, :, :]))
                targets_batch = np.vstack((targets_batch, targets_temp[np.newaxis, :, :]))

            # print(gru_input_batch.shape)
            # print(targets_batch.shape)
            np.random.shuffle(gru_input_batch)
            np.random.shuffle(targets_batch)
            np.savez(os.path.join(dataset_dir, os.pardir, 'new', f'{feature_dim}_dimensional_features.npz'),
                     gru_input_batch, targets_batch)
            batch = []
            max_step = 0

print('finish')
