import cv2
import numpy as np
import keras

from regression_model_lib import prepare_image
from regression_model_lib import load_imagenet_model


folder_name = 'test_video'
model_input_shape = (256, 256, 3)

video_file_name = 2

vid = cv2.VideoCapture(f'/mnt/external/{folder_name}/{video_file_name}.mp4')
detection = np.load(f'/mnt/external/{folder_name}_computed/{video_file_name}.npz')['arr_0']

imagenet_model = load_imagenet_model(input_shape=model_input_shape)

model = keras.models.load_model('./models/model.h5')

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

                bb = np.array(bb)
                bb = bb.astype(int)

                cropped_im, _ = prepare_image(im, bb)
                # cropped_im_bb = np.array(cropped_im_bb)
                feature_temp = imagenet_model.predict(cropped_im[np.newaxis, :, :, :])

                feature = feature_temp.reshape((feature_temp.shape[0], -1))

                # model.summary()

                result = model.predict(feature)
                result = result*cropped_im.shape[0]
                new_im = cv2.rectangle(cropped_im, (result[0, 0], result[0, 1]),
                                       (result[0, 0] + result[0, 2], result[0, 1] + result[0, 3]),
                                       color=(255, 0, 0), thickness=2)
                cv2.imshow('', cv2.resize(new_im, (800, 800)))
                cv2.waitKey(500)

