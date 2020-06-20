import numpy as np
import cv2
import os
import tensorflow.keras as k
import efficientnet.tfkeras as e
from utils import prepare_image


model_dir = 'fer_model_path'
fer_model = k.models.load_model(model_dir, compile=False)
# fer_model.compile(loss='mse',
#                   optimizer='adam')
layer_name = 'global_average_pooling2d'
intermediate_layer_model = k.models.Model(inputs=fer_model.input,
                                          outputs=fer_model.get_layer(layer_name).output)
intermediate_layer_model.summary()

root = os.path.abspath('dataset_path')
dataset_dir = os.path.join(root, 'cohn-kanade-images')
label_dir = os.path.join(root, 'Emotion')
landmark_dir = os.path.join(root, 'Landmarks')
subject_list = os.listdir(dataset_dir)
subject_label_list = os.listdir(label_dir)

c = 0
feature_dim = 1280
for subject in subject_list:
    print(subject)
    folder_list = os.listdir(os.path.join(dataset_dir, subject))
    try:
        folder_list.remove('.DS_Store')
    except ValueError:
        pass
    for folder in folder_list:
        frames_list = os.listdir(os.path.join(dataset_dir, subject, folder))
        landmark_list = os.listdir(os.path.join(landmark_dir, subject, folder))

        try:
            label_list = os.listdir(os.path.join(label_dir, subject, folder))
            if len(label_list) == 0:
                c += 1
                continue
        except FileNotFoundError:
            c += 1
            continue

        if not os.path.exists(f"dataset\\{subject}"):
            os.makedirs(f"dataset\\{subject}")

        try:
            frames_list.remove('.DS_Store')
        except ValueError:
            pass
        try:
            landmark_list.remove('.DS_Store')
        except ValueError:
            pass

        cropped_frames = []
        landmarks = []
        for frame in frames_list:
            frame_dir = os.path.join(dataset_dir, subject, folder, frame)
            im = cv2.imread(frame_dir)

            landmark = os.path.join(landmark_dir, subject, folder, f'{frame.split(".")[0]}_landmarks.txt')
            landmark_file = open(landmark)
            lines = landmark_file.readlines()

            landmark = []
            for line in lines:
                elem = line.strip()
                elem = elem.split('  ')
                landmark.append([float(i) for i in elem])

            landmarks.append(landmark)

            maximum = np.amax(landmark, axis=0)
            minimum = np.amin(landmark, axis=0)
            bb = np.concatenate([minimum, maximum], 0)

            new_im, _ = prepare_image(im, bb, shape=(224, 224), convert_shape=True)

            # cv2.imshow(f'{frame}', new_im)
            # cv2.waitKey(1)
            # cv2.destroyAllWindows()

            new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
            new_im = new_im.astype(np.float32)/255
            cropped_frames.append(new_im)

        label_file = open(os.path.join(label_dir, subject, folder, label_list[0]))
        label = label_file.read()
        label = label.strip()
        label = int(float(label))
        features = intermediate_layer_model.predict(np.asarray(cropped_frames))
        # print(features.shape)
        # print(label)

        path = os.path.join(f"dataset\\{subject}", f'{subject}_{folder}.npz')
        np.savez(path, features, np.atleast_1d(label), np.asarray(landmarks, dtype=np.float16))


print('finish')
