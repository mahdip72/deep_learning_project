import numpy as np
import matplotlib.pyplot as plt
import os

# from feature_extraction_lib import convert_bb_shape
from regression_model_lib import plot_model
from regression_model_lib import IoU

from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam


def regression_model(dir, model_name):

    names = os.listdir(dir)

    data = np.zeros((0, 1288))
    for name in names:
        data = np.concatenate((data, np.load(f'{dir}/{name}')), axis=0)

    bb_for_label = data[:, 0:4]
    bb_for_train = data[:, 4:8]
    features = data[:, 8:]

    del data

    # covert bounding boxes to [X, Y, W, H] format
    # bb_for_train = convert_bb_shape(bb_for_train)
    # bb_for_label = convert_bb_shape(bb_for_label)

    # change coordinate system in bounding boxes
    bb_for_label[:, 0] = bb_for_label[:, 0] - bb_for_train[:, 0]
    bb_for_label[:, 1] = bb_for_label[:, 1] - bb_for_train[:, 1]
    bb_for_train[:, 0] = 0
    bb_for_train[:, 1] = 0

    # normalize bounding boxes with width and height
    bb_for_label[:, 0] = np.divide(bb_for_label[:, 0], bb_for_train[:, 2])
    bb_for_label[:, 1] = np.divide(bb_for_label[:, 1], bb_for_train[:, 3])
    bb_for_label[:, 2] = np.divide(bb_for_label[:, 2], bb_for_train[:, 2])
    bb_for_label[:, 3] = np.divide(bb_for_label[:, 3], bb_for_train[:, 3])

    bb_for_train[:, 2] = np.divide(bb_for_train[:, 2], bb_for_train[:, 2])
    bb_for_train[:, 3] = np.divide(bb_for_train[:, 3], bb_for_train[:, 3])

    # create center point for input
    # center_point_x = bb_for_train[:, 0] + bb_for_train[:, 2]/2
    # center_point_y = bb_for_train[:, 1] + bb_for_train[:, 3]/2

    # reshape from 1-d to 2-d array
    # center_point_x = np.reshape(center_point_x, (-1, 1))
    # center_point_y = np.reshape(center_point_y, (-1, 1))

    # center_point_input = np.concatenate((center_point_x, center_point_y), axis=1)

    # create center point for label
    # center_point_x = bb_for_label[:, 0] + bb_for_label[:, 2]/2
    # center_point_y = bb_for_label[:, 1] + bb_for_label[:, 3]/2

    # reshape from 1-d to 2-d array
    # center_point_x = np.reshape(center_point_x, (-1, 1))
    # center_point_y = np.reshape(center_point_y, (-1, 1))

    # center_point_label = np.concatenate((center_point_x, center_point_y), axis=1)

    # create distance between two centers in from
    # distance_between_centers_x = center_point_input[:, 0] - center_point_label[:, 0]
    # distance_between_centers_y = center_point_input[:, 1] - center_point_label[:, 1]

    # input_data = np.concatenate((bb_for_train[:, 2:], features), axis=1)

    # target_data = np.concatenate((bb_for_label[:, 2:],
    #                               np.reshape(distance_between_centers_x, (-1, 1)),
    #                               np.reshape(distance_between_centers_y, (-1, 1))), axis=1)

    input_data = features
    target_data = bb_for_label

    # create model
    reg_model = Sequential()
    reg_model.add(Dense(512, activation='relu', input_shape=[input_data.shape[1]]))
    # reg_model.add(Dropout(0.5))
    reg_model.add(Dense(128, activation='relu'))
    # reg_model.add(Dropout(0.25))
    reg_model.add(Dense(32, activation='relu'))
    reg_model.add(Dense(target_data.shape[1], activation='linear'))

    reg_model.summary()

    reg_model.compile(optimizer=Adam(lr=0.01),
                      # loss='mse',
                      loss="mae",
                      # loss='mean_absolute_percentage_error',
                      metrics=[IoU])

    if not os.path.exists("./models/checkpoint/"):
        os.makedirs("./models/checkpoint/")

    # reg_model.load_weights(f'./models/checkpoint/checkpoint_{model_name}_model.h5')

    # checkpoint = ModelCheckpoint(f'./models/checkpoint/checkpoint_{model_name}_model.h5',
    #                              monitor='val_loss', verbose=0, save_best_only=True,
    #                              save_weights_only=True, mode='min', period=1)
    # callback = [checkpoint]

    history = reg_model.fit(input_data, target_data,
                            validation_split=0.05,
                            epochs=20, batch_size=512, verbose=2,
                            # callbacks=callback,
                            shuffle=True)

    if not os.path.exists("./models"):
        os.makedirs("./models")

    reg_model.save(f'./models/{model_name}_model.h5')

    plot_model(history)

    return reg_model


if __name__ == '__main__':
    """
    before train regression model the extract features part must be executed
    """

    model = regression_model(dir="/mnt/external/test_video_features", model_name="test_video_dataset")
    print('done')

