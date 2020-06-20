import numpy as np
import cv2
import tensorflow as tf
import pandas as pd
import os
import time

from regression_model_lib import load_imagenet_model
from regression_model_lib import create_id_model
from regression_model_lib import extract_id_features
from regression_model_lib import prepare_image
from regression_model_lib import compute_detectron_detection

from regression_model_lib import split_image_into_tiles
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input


def compute_id_features():
    """
    extract identity features with duke network from stanford dataset
    :param frame: input frame
    :param bb: bounding boxes [X1, Y2, W, H]
    :param test_video: for testing in a video file
    :return: features
    """

    # size of input network
    net_input_size = (256, 128)
    cropped_im = np.zeros((0, 256, 128, 3))
    features = np.zeros((0, 128))

    # create placeholder for input images with shape of (None, 256, 128, 3)
    # None refers to the number of images in batch
    images_in_tensor = tf.placeholder(tf.float32, shape=(None, net_input_size[0], net_input_size[1], 3))

    # create model
    checkpoints, endpoints, config = create_id_model(images_in_tensor)

    df = pd.read_csv('./preprocess/new_test.csv')

    counter = 0
    for i in range(len(df)):
        # for i in range(250):
        image_name = df.iloc[i, 0]
        frame = cv2.imread(f'/home/mehdi/Desktop/Dataset/Stanford40/JPEGImages/{image_name}.jpg')
        bb = np.array(df.iloc[i, 1:5]).reshape((1, 4))
        bb = bb.astype(int)

        cropped_im_temp = frame[bb[0, 1]:bb[0, 1] + bb[0, 3], bb[0, 0]:bb[0, 0] + bb[0, 2], :]
        cropped_im_temp = cv2.resize(cropped_im_temp, net_input_size[::-1])
        cropped_im = np.vstack((cropped_im, cropped_im_temp[np.newaxis, :, :, :]))

        counter += 1
        # print(counter)

        if counter == 250:
            counter = 0
            # extract features
            features_temp = extract_id_features(checkpoints, endpoints, cropped_im, config, images_in_tensor)
            print('batch size:', features_temp.shape[0])
            features = np.vstack((features, features_temp))
            cropped_im = np.zeros((0, 256, 128, 3))

        # plot cropped image
        # cv2.imshow('show', cropped_im_temp)
        # cv2.waitKey(0)

    if counter != 0:
        # extract features
        features_temp = extract_id_features(checkpoints, endpoints, cropped_im, config, images_in_tensor)
        print('batch size:', features_temp.shape[0])
        features = np.vstack((features, features_temp))

    new_df = pd.DataFrame(features)
    print('create features done!')

    return new_df


def compute_imagenet_backbone_features(model_input_shape=(256, 256, 3), dir='/mnt/external'):
    """

    :param model_input_shape:
    :param dir: correct : '/mnt/external/7' not '/mnt/external/7/'
    :return:
    """
    imagenet_model, features_shape = load_imagenet_model(input_shape=model_input_shape,
                                                         return_feature_shape=True)

    folder_name = dir.split(sep='/')[-1]
    folder_dir = '/'.join(dir.split(sep='/')[:-1])

    dirs = os.listdir(folder_dir + f"/{folder_name}/")

    if not os.path.exists(folder_dir + f"/{folder_name}_features/"):
        os.makedirs(folder_dir + f"/{folder_name}_features/")

    computed_dirs = os.listdir(folder_dir + f"/{folder_name}_features/")

    # remove extension in file names (mp4, npz)
    dirs = [i[:-4] for i in dirs]
    computed_dirs = [i[9:-4] for i in computed_dirs]

    # subtract computed files
    dirs = set(dirs)
    computed_dirs = set(computed_dirs)
    dirs = list(dirs - computed_dirs)

    for i in dirs:
        print(f'start processing of {i}')
        cropped_im = np.zeros((0, 256, 256, 3))
        features = np.zeros((0, features_shape[-1]))
        bbs = np.zeros((0, 8))

        detection = np.load(folder_dir + f"/{folder_name}_computed/{i}.npz")['arr_0']
        vid = cv2.VideoCapture(folder_dir + f"/{folder_name}/{i}.mp4")

        frame_number = 0
        counter = 0
        success = True
        while success:
            success, im = vid.read()
            if success:
                frame_number += 1
                in_frame_list = detection[detection[:, 0] == frame_number][:, 1:].tolist()
                if len(in_frame_list) > 0:
                    for bb in in_frame_list:
                        bb = np.array(bb).astype(int)
                        # bb = bb.astype(int)

                        cropped_im_temp, cropped_im_bb = prepare_image(im, bb)
                        # cropped_im_bb = np.array([0, 0, 256, 256])
                        bbs_temp = np.concatenate((bb, cropped_im_bb), axis=0)
                        bbs = np.vstack((bbs, bbs_temp[np.newaxis, :]))

                        # show cropped image
                        # cv2.imshow('cropped image', cropped_im_temp)
                        # cv2.waitKey(100)

                        cropped_im = np.vstack((cropped_im, cropped_im_temp[np.newaxis, :, :, :]))
                        counter += 1

                        if counter == 250:
                            counter = 0
                            # extract features from imagenet model
                            features_temp = imagenet_model.predict(cropped_im)
                            features_temp = features_temp.reshape((features_temp.shape[0], -1))
                            print('batch size:', features_temp.shape[0])
                            features = np.vstack((features, features_temp))
                            cropped_im = np.zeros((0, 256, 256, 3))
                else:
                    continue

        if counter != 0:
            features_temp = imagenet_model.predict(cropped_im)
            features_temp = features_temp.reshape((features_temp.shape[0], -1))
            print('batch size:', features_temp.shape[0])
            features = np.vstack((features, features_temp))

        features = np.concatenate((bbs, features), axis=1)

        if not os.path.exists(folder_dir + f"/{folder_name}_features/"):
            os.makedirs(folder_dir + f"/{folder_name}_features/")
        np.save(folder_dir + f"/{folder_name}_features/features_{i}.npy", features)

        print(f'feature processing for {i} done')
        print('.........................................................')

        # start = time.time()
        # batch_of_tiles = split_image_into_tiles(cropped_im)
        # end = time.time()
        # print(f'splitting image into tiles takes {(end - start) * 1000:.2f} milli sec')

        # features = create_features_for_images(imagenet_model, batch_of_tiles)

    print('extracting features completed')


if __name__ == '__main__':
    """
    given directory of video files
    1- extracting and saving detectron detection
    2- extracting and saving features with pre train imagenet model
    
    """

    compute_detectron_detection(dir='/mnt/external/test_video', gpu='0')
    compute_imagenet_backbone_features(dir='/mnt/external/test_video')

    # data_id_features = compute_id_features()
    # print('number of sample: ', len(data_features))

