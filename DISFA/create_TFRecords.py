import tensorflow as tf
import numpy as np
import os
import random
import pandas as pd
import scipy.io
import cv2
import pickle
from keras.utils.np_utils import to_categorical


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def prepare_data(data):
    s = np.asarray(data)

    x = s[:, 0]
    y = {'AU1': np.asarray([i['AU1'] for i in s[:, 1]]),
         'AU2': np.asarray([i['AU2'] for i in s[:, 1]]),
         'AU4': np.asarray([i['AU4'] for i in s[:, 1]]),
         'AU5': np.asarray([i['AU5'] for i in s[:, 1]]),
         'AU6': np.asarray([i['AU6'] for i in s[:, 1]]),
         'AU9': np.asarray([i['AU9'] for i in s[:, 1]]),
         'AU12': np.asarray([i['AU12'] for i in s[:, 1]]),
         'AU15': np.asarray([i['AU15'] for i in s[:, 1]]),
         'AU17': np.asarray([i['AU17'] for i in s[:, 1]]),
         'AU20': np.asarray([i['AU20'] for i in s[:, 1]]),
         'AU25': np.asarray([i['AU25'] for i in s[:, 1]]),
         'AU26': np.asarray([i['AU26'] for i in s[:, 1]])}
    return x, y


def get_images_dirs(sub_dirs):
    images_dir = []
    for folder, label in sub_dirs:
        label_list = []
        keys = list(label.keys())
        images_list = os.listdir(folder)
        images_list.remove('TimeStamp.txt')
        for image_indx in range(len(images_list)):
            au = dict()
            for key in keys:
                # sparse categorical:
                # au[key] = np.array([label[key][image_indx]])
                # categorical:
                au[key] = np.squeeze(to_categorical([label[key][image_indx]], num_classes=6))
            label_list.append(au)
        for indx, image in enumerate(images_list):
            images_dir += [[os.path.join(folder, image), label_list[indx]]]
    return images_dir


def load_aus(label_dir):
    aus = dict()
    file_list = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26']
    for file in file_list:
        vars()[file] = pd.read_csv(os.path.join(label_dir, file + '.txt'))
        vars()[file] = vars()[file].iloc[1:, 0]
        vars()[file] = vars()[file].tolist()
        vars()[file] = [int(item[-1]) for item in vars()[file]]
        aus[file] = vars()[file]
    return aus


def calculate_class_unbalancing(data):
    class_weights = dict()
    keys = ['AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU9', 'AU12', 'AU15', 'AU17', 'AU20', 'AU25', 'AU26']
    # create dict for each AU:
    for key in keys:
        vars()[key] = np.zeros((6,))
    # counting class sample for each AU
    for image_dir, label in data:
        for key in keys:
            vars()[key] += label[key]

    # for key in keys:
    #     maximum = np.max(vars()[key])
    #     for i, j in enumerate(vars()[key]):
    #         if j != 0:
    #             vars()[key][i] = maximum / j
    #         else:
    #             vars()[key][i] = 1

    for key in keys:
        au = {}
        for i, j in enumerate(vars()[key]):
            au[i] = j
        class_weights[key] = au

    return class_weights


class DataPreprocess:
    def __init__(self, data_list, landmarks_path, dataset_path, labels_path, shuffle=False):
        self.data_list = data_list
        self.data = []
        self.landmarks_path = landmarks_path
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.shuffle = shuffle

    def get_sub_dirs(self, subject_list):
        sub_dirs_and_labels = []
        for subject in subject_list:
            # for image directory
            d = os.path.join(self.dataset_path, subject)
            # for label directory
            l = os.path.join(self.labels_path, subject)
            sub_dirs_and_labels = sub_dirs_and_labels + [[os.path.join(d, o), load_aus(os.path.join(l, o))] for o in
                                                         os.listdir(d)
                                                         if os.path.isdir(os.path.join(d, o))]
        return get_images_dirs(sub_dirs_and_labels)

    def load_landmark(self, directory):
        image_name = os.path.split(directory)[-1]
        folder_name = os.path.split(os.path.split(directory)[0])[-1]
        subject_name = os.path.split(os.path.split(os.path.split(directory)[0])[0])[-1]

        mat = scipy.io.loadmat(os.path.join(self.landmarks_path, subject_name, folder_name + '_FaceCropped.mat'))
        for row in mat['FaceImg_CropResize'][0]:
            if row[0][0][0][0] == image_name:
                return row[0][0][1]


def image_reshape(img, label):
    img = tf.reshape(img, [224, 224, 3])
    return img, label


model_name = 'efficientb0_ccc_loss_extra_layer_2'
save_path = os.path.abspath('D:\mehdi\models\AffectNet')
model_path = os.path.join(save_path, f"{model_name}")

# if not os.path.exists(model_path):
#     os.makedirs(model_path)

auto = tf.data.experimental.AUTOTUNE
batch = 32

data_path = os.path.abspath('E:\Document\Database\FER\DISFA+\Images')
landmark_path = os.path.abspath('E:\Document\Database\FER\DISFA+\FaceLandmarks\landmarks')
label_path = os.path.abspath('E:\Document\Database\FER\DISFA+\Labels')
lst = os.listdir(data_path)

directories = []
for i in lst:
    if os.path.isdir(os.path.join(data_path, i)):
        directories.append(i)


def load_cropped_image(path, landmarks_path=landmark_path):
    directory = os.path.abspath(str(path.numpy()))
    # directory = os.path.abspath(path)s
    image_name = os.path.split(directory)[-1]
    folder_name = os.path.split(os.path.split(directory)[0])[-1]
    subject_name = os.path.split(os.path.split(os.path.split(directory)[0])[0])[-1]

    mat = scipy.io.loadmat(os.path.join(landmarks_path, subject_name, folder_name + '_FaceCropped.mat'))
    cropped_img = mat['FaceImg_CropResize'][0, int(image_name.split('.')[0])][0, 0][1]
    cropped_img = cv2.resize(cropped_img, (224, 224))
    return cropped_img / 255


def serialize_example(image, AUs):
    feature = {
        'image': _bytes_feature(image),
        'AU1': _bytes_feature(tf.io.serialize_tensor(AUs['AU1'])),
        'AU2': _bytes_feature(tf.io.serialize_tensor(AUs['AU2'])),
        'AU4': _bytes_feature(tf.io.serialize_tensor(AUs['AU4'])),
        'AU5': _bytes_feature(tf.io.serialize_tensor(AUs['AU5'])),
        'AU6': _bytes_feature(tf.io.serialize_tensor(AUs['AU6'])),
        'AU9': _bytes_feature(tf.io.serialize_tensor(AUs['AU9'])),
        'AU12': _bytes_feature(tf.io.serialize_tensor(AUs['AU12'])),
        'AU15': _bytes_feature(tf.io.serialize_tensor(AUs['AU15'])),
        'AU17': _bytes_feature(tf.io.serialize_tensor(AUs['AU17'])),
        'AU20': _bytes_feature(tf.io.serialize_tensor(AUs['AU20'])),
        'AU25': _bytes_feature(tf.io.serialize_tensor(AUs['AU25'])),
        'AU26': _bytes_feature(tf.io.serialize_tensor(AUs['AU26'])),
    }
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


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


for subject in directories:
    subject = [subject]

    # subject_process = DataPreprocess(data_list=subject,
    #                                  landmarks_path=landmark_path,
    #                                  dataset_path=data_path,
    #                                  labels_path=label_path,
    #                                  shuffle=True)
    # subject_process.data = subject_process.get_sub_dirs(subject_list=subject_process.data_list)

    # class_weights = calculate_class_unbalancing(subject_process.data)

    # with open(os.path.join(data_path, subject[0] + ".pkl"), "wb") as a_file:
    #     pickle.dump(class_weights, a_file)

    with open("E:\\gik\\" + subject[0] + ".pkl", "rb") as a_file:
        output = pickle.load(a_file)
    #     print(output)

    # random.shuffle(subject_process.data)
    # subject_process.data = prepare_data(subject_process.data)
    # ds = tf.data.Dataset.from_tensor_slices(subject_process.data)
    # ds = ds.map(map_func=image_reshape, num_parallel_calls=auto).prefetch(auto)

    # for i, j in ds:
    #     print(i.shape)
    #     print(j)

    # file_path = os.path.join(model_path, subject[0] + '.tfrecords')
    # with tf.io.TFRecordWriter(file_path) as writer:
    #     for image, labels in ds:
    #         serialized_example = serialize_example(tf.io.serialize_tensor(image),
    #                                                labels)
    #         writer.write(serialized_example)
    #
    print(subject[0], 'done')

print('finish')
