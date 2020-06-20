import numpy as np
import cv2
import tensorflow as tf
import traceback
import os
import matplotlib.pyplot as plt

from importlib import import_module
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from skimage.util.shape import view_as_blocks

from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.nasnet import NASNetMobile
# from keras.applications.nasnet import preprocess_input
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.xception import Xception
# from keras.applications.xception import preprocess_input


def detectron_model(detectron_dir='/home/mehdi/mehdi/other', gpu='0', score_thresh=0.9):
    """
    load detectron model base on detection config
    :param detectron_dir: directory of detectron model
    :param gpu:
    :param score_thresh: confidence threshold for detection
    :return:
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = f'cuda:{gpu}'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    # cfg.merge_from_file(detectron_dir + "/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.merge_from_file("./detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    # cfg.merge_from_file(detectron_dir + "/detectron2/configs/COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.merge_from_file(detectron_dir + "/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    # cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    # cfg.merge_from_file("./detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    # cfg.MODEL.WEIGHTS = detectron_dir + "/models/segmentation/best_acc_139653917.pkl"
    # cfg.MODEL.WEIGHTS = "./models/segmentation/optimize_137849600.pkl"
    # cfg.MODEL.WEIGHTS = detectron_dir + "/models/keypoints/best_acc_139686956.pkl"
    cfg.MODEL.WEIGHTS = detectron_dir + "/models/keypoints/optimize_137849621.pkl"
    # cfg.MODEL.WEIGHTS = "./models/object_detection/best_acc_139173657.pkl"
    # cfg.MODEL.WEIGHTS = "./models/object_detection/optimize_137849458.pkl"

    predictor = DefaultPredictor(cfg)

    return predictor, cfg


def compute_detectron_detection(dir='/mnt/external/7', gpu='0', show_result=False):
    """
    :param dir: directory of mp4 video files ,correct : '/mnt/external/7'
                                              incorrect:'/mnt/external/7/'
    :param gpu: gpu number
    :param show_result: plot detection result
    :return:
    """

    predictor, cfg = detectron_model(gpu=gpu, score_thresh=0.9)

    folder_name = dir.split(sep='/')[-1]
    folder_dir = '/'.join(dir.split(sep='/')[:-1])

    dirs = os.listdir(folder_dir + f"/{folder_name}/")

    if not os.path.exists(folder_dir + f"/{folder_name}_computed/"):
        os.makedirs(folder_dir + f"/{folder_name}_computed/")

    computed_dirs = os.listdir(folder_dir + f"/{folder_name}_computed/")

    # remove extension in file names
    dirs = [i[:-4] for i in dirs]
    computed_dirs = [i[:-4] for i in computed_dirs]

    # subtract computed files
    dirs = set(dirs)
    computed_dirs = set(computed_dirs)
    dirs = list(dirs - computed_dirs)

    for i in dirs:
        vid = cv2.VideoCapture(folder_dir + f"/{folder_name}/{i}.mp4")
        print(f'{i} processing start')

        kp = np.zeros((0, 52))
        bb = np.zeros((0, 5))

        frame_number = 0
        success = True
        while success:
            success, im = vid.read()
            if success:
                frame_number += 1
                output = predictor(im)

                key_points = np.zeros(shape=(output['instances'].get('pred_keypoints').shape[0], 52))
                bonding_boxes = np.zeros(shape=(output['instances'].get('pred_boxes').tensor.shape[0], 5))

                # Check human is detected or not for extract features
                if len(output['instances'].get('pred_keypoints')) <= 0:
                    key_points = np.empty((0, 52))
                    bonding_boxes = np.empty((0, 5))
                else:
                    # extracting key points from output
                    key_points[:, 0] = frame_number
                    key_points[:, 1:] = output['instances'].get('pred_keypoints').cpu().numpy().reshape(key_points.shape[0], 51)

                    # extracting bonding boxes from output
                    bonding_boxes[:, 0] = frame_number
                    bonding_boxes[:, 1:] = output['instances'].get('pred_boxes').tensor.cpu().numpy()
                    bonding_boxes[:, 3] = np.abs(bonding_boxes[:, 1] - bonding_boxes[:, 3])
                    bonding_boxes[:, 4] = np.abs(bonding_boxes[:, 2] - bonding_boxes[:, 4])

                kp = np.vstack((kp, key_points))
                bb = np.vstack((bb, bonding_boxes))

                # if frame_number % 100 == 0:
                #     print('number of processed frames: ', frame_number)

                if show_result:
                    # show result image
                    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
                    v = v.draw_instance_predictions(output["instances"].to("cpu"))
                    image = v.get_image()[:, :, ::-1]
                    cv2.imshow('image', cv2.resize(image, (960, 540)))
                    cv2.waitKey(30)
        print('------------------------------')

        print(f'{i} processing complete')
        np.savez(folder_dir + f'/{folder_name}_computed/{i}.npz', bb, kp)

    print('------------------------------')
    print('detection processing completed')


def convert_bb_shape(bb):
    """

    :param bb: 2-d array format (x1, y1, x2, y2)
    :return: 2-d array format (x, y, w, h)
    """

    bb[:, 2] = np.abs(bb[:, 0] - bb[:, 2])
    bb[:, 3] = np.abs(bb[:, 1] - bb[:, 3])

    return bb


def plot_circle(im, center):
    color = (0, 0, 255)
    img = cv2.circle(im, (int(center[0]), int(center[1])), radius=12, color=color, thickness=-1)
    cv2.imshow('', cv2.resize(img, (int(im.shape[1]/3), int(im.shape[0]/3))))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()


def load_imagenet_model(input_shape=(256, 256, 3), return_feature_shape=False):
    model = MobileNetV2(weights='imagenet',
                        input_shape=input_shape,
                        pooling='avg',
                        include_top=False)

    # model.summary()
    if return_feature_shape:
        return model, model.output.shape
    else:
        return  model


def extract_id_features(checkpoint, model, input_images, configs, images_in_tensor, net_input_size=(256, 128)):

    # resize input images to (256, 128, 3) shape
    for j in range(input_images.shape[0]):
        input_images[j, :, :, :] = cv2.resize(input_images[j, :, :, :], net_input_size[::-1])

    # with tf.Session(config=tf.ConfigProto(use_per_session_threads=True)) as sess:
    with tf.Session(config=configs) as sess:
        tf.train.Saver().restore(sess, checkpoint)
        # print('f1')
        id_features = sess.run(model['emb'], feed_dict={images_in_tensor: input_images})
        # print('f2')
    return id_features


def create_features_for_images(model, images):
    features = model.predict(images)
    print('features shape for input images:', features.shape)
    return features


def create_id_model(images_in_tensor, checkpoints_dir='/home/mehdi/mehdi/Mehdi_dTrack/', model_dir='src.triplet_reid'):

    # create model and head: convolution resnet and full connected with 1024 neuron
    model = import_module(model_dir + '.nets.resnet_v1_50')
    head = import_module(model_dir + '.heads.fc1024')

    # image_in_tensor = tf.convert_to_tensor(images, dtype=tf.float32)

    # create outputs of model
    endpoints, model_name = model.endpoints(images_in_tensor, is_training=False)

    with tf.name_scope('head'):
        endpoints = head.head(endpoints, 128, is_training=False)

    # load weights of created model
    checkpoint = tf.train.latest_checkpoint(checkpoints_dir + 'demo_weighted_triplet/')

    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.use_per_session_threads = True

    return checkpoint, endpoints, config


def split_image_into_tiles(image, block_shape=(64, 64, 3)):
    """

    :param image: input image
    :param block_shape: splitting image to multiple image with shape of block_shape
    :return: 4-d array of splitted images
    """
    blocks = view_as_blocks(image, block_shape=block_shape)
    print('block shape for splitting image:', block_shape)

    images = np.zeros(shape=(0,)+block_shape)
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            tile = blocks[i, j, 0, :, :, :]

            # showing tiles
            cv2.imshow('', cv2.resize(tile, (128, 128)))
            cv2.waitKey(500)

            tile = np.expand_dims(tile, axis=0)
            images = np.concatenate((images, tile), axis=0)
    print(f'image splits into {images.shape[0]} tiles')
    cv2.destroyAllWindows()
    return images


def prepare_image(image, bb):
    """

    :param image:
    :param bb:
    :return: extracted square shape x-box with respect to bb, size of x-box before resizing to 256x256
    """
    check_bb = bb.copy()
    # bb[2] = int(bb[2]) - int(bb[0])
    # bb[3] = int(bb[3]) - int(bb[1])

    # side of the square
    # s = int(max(bb[2], bb[3]))
    # s = int((bb[2] + bb[3])/2)
    s = int(max(bb[2], bb[3])*np.random.uniform(low=0.6, high=1))

    height = image.shape[0]
    width = image.shape[1]

    # center of bb : (Cx, Cy)
    center = [bb[0]+bb[2]/2, bb[1]+bb[3]/2]

    center[0] = int(center[0])
    center[1] = int(center[1])

    # plot_circle(image, center)

    # check side of the square is not bigger than frame height and frame width
    if 2*s > width or 2*s > height:
        s = int(min(width, height)/2)

    if (center[0] - s) < 0:
        center[0] = center[0] + abs(center[0] - s)
        # plot_circle(image, center)

    if (center[0] + s) > width:
        center[0] = center[0] - (center[0] + s - width)
        # plot_circle(image, center)

    if (center[1] - s) < 0:
        center[1] = center[1] + abs(center[1] - s)
        # plot_circle(image, center)

    if (center[1] + s) > height:
        center[1] = center[1] - (center[1] + s - height)
        # plot_circle(image, center)

    new_image = image[center[1]-s:center[1]+s, center[0]-s:center[0]+s]

    h = new_image.shape[0]
    w = new_image.shape[1]

    # cv2.imshow('new image', cv2.resize(image, (900, 900), interpolation=cv2.INTER_NEAREST))
    # cv2.imshow('new image', image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    # start = time.time()
    try:
        new_image = cv2.resize(new_image, (256, 256), interpolation=cv2.INTER_CUBIC)
    except Exception as e:
        print(check_bb)
        cv2.imshow('new image', new_image)
        cv2.waitKey()
        traceback.print_exc()

    # end = time.time()
    # print(f'resizing image takes {(end-start)*1000:.2f} milli sec')

    return new_image, [center[0]-s, center[1]-s, w, h]


def plot_bb(bb, image):
    pass


def plot_model(history):
    """
    plot loss function for a trained model
    :param history:
    :return:
    """
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Regression Models loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def calculate_iou(y_true, y_pred):
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).

    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height]
    y_pred -- second box, numpy array with format [x, y, width, height]
    x any y are the coordinates of the top left corner of each box.

    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)

    """

    results = []

    for i in range(0, y_true.shape[0]):
        # set the types so we are sure what type we are using
        y_true = y_true.astype(np.float32)
        y_pred = y_pred.astype(np.float32)

        # boxTrue
        x_boxtrue_tleft = y_true[0, 0]  # numpy index selection
        y_boxtrue_tleft = y_true[0, 1]
        boxtrue_width = y_true[0, 2]
        boxtrue_height = y_true[0, 3]
        area_boxtrue = (boxtrue_width * boxtrue_height)

        # boxPred
        x_boxpred_tleft = y_pred[0, 0]
        y_boxpred_tleft = y_pred[0, 1]
        boxpred_width = y_pred[0, 2]
        boxpred_height = y_pred[0, 3]
        area_boxpred = (boxpred_width * boxpred_height)

        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxtrue_br = x_boxtrue_tleft + boxtrue_width
        y_boxtrue_br = y_boxtrue_tleft + boxtrue_height  # Version 2 revision

        # boxPred
        x_boxpred_br = x_boxpred_tleft + boxpred_width
        y_boxpred_br = y_boxpred_tleft + boxpred_height  # Version 2 revision

        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxint_tleft = np.max([x_boxtrue_tleft, x_boxpred_tleft])
        y_boxint_tleft = np.max([y_boxtrue_tleft, y_boxpred_tleft])  # Version 2 revision

        # boxInt - bottom right coords
        x_boxint_br = np.min([x_boxtrue_br, x_boxpred_br])
        y_boxint_br = np.min([y_boxtrue_br, y_boxpred_br])

        # Calculate the area of boxInt, i.e. the area of the intersection
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.

        # Version 2 revision
        area_of_intersection = \
            np.max([0, (x_boxint_br - x_boxint_tleft)]) * np.max([0, (y_boxint_br - y_boxint_tleft)])

        iou = area_of_intersection / ((area_boxtrue + area_boxpred) - area_of_intersection)

        # This must match the type used in py_func
        iou = iou.astype(np.float32)

        # append the result to a list at the end of each loop
        results.append(iou)

    # return the mean IoU score for the batch
    return np.mean(results)


def IoU(y_true, y_pred):
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours
    # trying to debug and almost give up.

    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou
