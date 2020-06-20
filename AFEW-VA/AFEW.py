import cv2
import numpy as np
import os
import json
import tensorflow.keras as k
from lib import FaceDetector
from keras import backend as K
from lib import prepare_image
from lib import r2

np.random.seed(10)
# MODEL_PATH = '../demo/model/Face_Detection/model.pb'
# face_detector = FaceDetector(MODEL_PATH)

fer_model_path = '../demo/model/AffectNet/affectnet.h5'
fer_model = k.models.load_model(fer_model_path)
layer_name = 'global_average_pooling2d_1'
intermediate_layer_model = k.models.Model(inputs=fer_model.input,
                                          outputs=fer_model.get_layer(layer_name).output)
intermediate_layer_model.summary()

x1 = k.layers.Input(shape=(None, 1280))
x1_drop = k.layers.Dropout(0.3)(x1)
x2 = k.layers.Bidirectional(k.layers.GRU(80, activation='relu'))(x1_drop)
x2 = k.layers.RepeatVector(K.shape(x1)[1])(x2)
x3 = k.layers.Bidirectional(k.layers.GRU(80, activation='relu', return_sequences=True))(x2)
x2 = k.layers.TimeDistributed(k.layers.Dense(2, activation='softmax'))(x2)
x4 = k.layers.Dense(32, activation='relu')(x3)
x5 = k.layers.Dense(2, activation='linear')(x4)

fer_gru_model = k.models.Model(x1, x5)

# fer_gru_model = k.models.Sequential([
#     # k.layers.InputLayer(input_shape=(32, 1280)),
#     # k.layers.Dropout(0.3),
#     k.layers.Bidirectional(k.layers.GRU(50, activation='relu', return_state=True, input_shape=(32, 1280))),
#     k.layers.RepeatVector(32),
#     k.layers.Bidirectional(k.layers.GRU(50, activation='relu', return_sequences=True)),
#     k.layers.TimeDistributed(k.layers.Dense(32, activation='relu')),
#     k.layers.Dense(2, activation='linear'),
# ])

# fer_gru_model.build(input_shape=(None, 32, 1280))
fer_gru_model.summary()
fer_gru_model.compile(optimizer=k.optimizers.Adam(lr=0.001), loss='mse', metrics=[r2])

dataset_dir = 'E:\Document\Database\FER\AFEW-VA\dataset'
epochs = 10
max_step = 0
batch = []
for epoch in range(epochs):
    print(f'start epoch {epoch + 1}')
    his = [0, 0]
    state = True
    sample = 0

    folders_dir = os.listdir(dataset_dir)
    for folder in folders_dir:
        subjects_dir = os.listdir(os.path.join(dataset_dir, folder))
        try:
            subjects_dir.remove('README.md')
        except:
            pass
        for subject in subjects_dir:
            sample += 1
            frames = os.listdir(os.path.join(dataset_dir, folder, subject))
            frames.remove(subject + '.json')
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
                new_im, _ = prepare_image(im, bb, shape=(224, 224), convert_shape=True)

                input_images = np.vstack((input_images, new_im[np.newaxis, :, :, :]))

                targets = np.vstack((targets, np.array([[valence, arousal]])))

            input_images = input_images / 255
            gru_input = intermediate_layer_model.predict(input_images)
            batch.append([gru_input, targets])
            # gru_input = strided_axis0(gru_input, max_time_steps)
            # targets = strided_axis0(targets/10, max_time_steps)
            if len(batch) == 50 or sample == 350:
                batch_size = 0
                targets_batch = np.zeros((0, max_step, 2))
                gru_input_batch = np.zeros((0, max_step, 1280))
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
                np.savez(os.path.join(dataset_dir, os.pardir, 'new', folder + '.npz'),
                         gru_input_batch, targets_batch)
                # his = fer_gru_model.train_on_batch(gru_input_batch, targets_batch, reset_metrics=state)
                state = False
                batch = []
                max_step = 0

    print('loss:', his[0], ',R2:', his[1])

print('finish')
