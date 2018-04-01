import numpy as np
import os
import cv2

from glob import glob
from tqdm import tqdm
from scipy.io import loadmat
from PIL import Image

DATA_PATH = '/media/tulan/Linux/Pycharm/Image_Edge/data/'
ROOT_PATH = '/media/tulan/Linux/Pycharm/Image_Edge/'
GROUND_PATH = DATA_PATH + '/groundTruth/'
ORI_IMAGE_PATH = DATA_PATH + '/images/'


def get_y_ori():
    y_train = []
    y_test = []
    y_val = []

    for foldname in os.listdir(GROUND_PATH):
        for img in sorted(os.listdir(os.path.join(GROUND_PATH, foldname))):
            IMG_DIR = os.path.join(GROUND_PATH, foldname, img)
            mat = loadmat(IMG_DIR)
            matdata = mat['groundTruth']

            # rotate to the same size
            data = matdata[0, 0]['Boundaries'][0, 0]
            if data.shape == (481, 321):  # size: H - W
                data = cv2.resize(data, (320, 480))  # size: W - H
                data = Image.fromarray(data)
                data = np.asarray(data.rotate(90, expand=True))
            else:
                data = cv2.resize(data, (480, 320))
            if foldname == 'test':
                y_test.append(data)
            elif foldname == 'val':
                y_val.append(data)
            else:
                y_train.append(data)

    y_train = np.asarray(y_train).reshape((-1, 320, 480, 1))
    y_test = np.asarray(y_test).reshape((-1, 320, 480, 1))
    y_val = np.asarray(y_val).reshape((-1, 320, 480, 1))
    np.save(DATA_PATH + 'y_train_ori.npy', y_train)
    np.save(DATA_PATH + 'y_test_ori.npy', y_test)
    np.save(DATA_PATH + 'y_val_ori.npy', y_val)


def get_y_concat():
    y_train = []
    y_test = []
    y_val = []

    for foldname in os.listdir(GROUND_PATH):
        for img in sorted(os.listdir(os.path.join(GROUND_PATH, foldname))):
            IMG_DIR = os.path.join(GROUND_PATH, foldname, img)
            mat = loadmat(IMG_DIR)
            matdata = mat['groundTruth']
            if matdata[0, 0]['Boundaries'][0, 0].shape == (321, 481):
                data_c = np.zeros((321, 481))
            else:
                data_c = np.zeros((481, 321))
            # data = matdata[0, 0]['Boundaries'][0, 0]
            # rotate to the same size
            for i in range(0, matdata.shape[1]):
                data = matdata[0, i]['Boundaries'][0, 0]
                data_c = np.maximum(data, data_c)
                if i == matdata.shape[1] - 1:
                    if data_c.shape == (481, 321):              # size: H - W
                        data_c = cv2.resize(data, (320, 480))   # size: W - H
                        data_c = Image.fromarray(data_c)
                        data_c = np.asarray(data_c.rotate(90, expand=True))
                    else:
                        data_c = cv2.resize(data_c, (480, 320))
                    if foldname == 'test':
                        y_test.append(data_c)
                    elif foldname == 'val':
                        y_val.append(data_c)
                    else:
                        y_train.append(data_c)

    y_train = np.asarray(y_train).reshape((-1, 320, 480, 1))
    y_test = np.asarray(y_test).reshape((-1, 320, 480, 1))
    y_val = np.asarray(y_val).reshape((-1, 320, 480, 1))
    np.save(DATA_PATH + 'y_train_concat.npy', y_train)
    np.save(DATA_PATH + 'y_test_concat.npy', y_test)
    np.save(DATA_PATH + 'y_val_concat.npy', y_val)


def get_images():
    X_train = []
    X_test = []
    X_val = []

    for foldname in tqdm(os.listdir(ORI_IMAGE_PATH)):
        i = 0
        for img in sorted(glob(ORI_IMAGE_PATH + '/' + foldname + '/*')):
            data = cv2.imread(img)
            if data.shape == (481, 321, 3):
                data = cv2.resize(data, (320, 480))
                data = Image.fromarray(data)
                data = np.asarray(data.rotate(90, expand=True))
            else:
                data = cv2.resize(data, (480, 320))
            if foldname == 'test':
                X_test.append(data)
            elif foldname == 'val':
                X_val.append(data)
            else:
                X_train.append(data)
            i += 1

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    X_val = np.asarray(X_val)
    np.save(DATA_PATH + 'X_train_ori.npy', X_train)
    np.save(DATA_PATH + 'X_test_ori.npy', X_test)
    np.save(DATA_PATH + 'X_val_ori.npy', X_val)


print('Load Labels')
get_y_ori()
get_y_concat()
print('load images')
get_images()
print('Finished Data Loading')

