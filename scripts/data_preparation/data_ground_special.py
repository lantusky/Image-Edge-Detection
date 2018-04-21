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


def get_y_moreThanHalf():
    y_train = []
    y_test = []
    y_val = []

    for foldname in tqdm(os.listdir(GROUND_PATH)):
        for img in sorted(os.listdir(os.path.join(GROUND_PATH, foldname))):
            IMG_DIR = os.path.join(GROUND_PATH, foldname, img)
            mat = loadmat(IMG_DIR)
            matdata = mat['groundTruth']
            if matdata[0, 0]['Boundaries'][0, 0].shape == (321, 481):
                count = np.zeros((321, 481), dtype=np.float32)
            else:
                count = np.zeros((481, 321), dtype=np.float32)
            # data = matdata[0, 0]['Boundaries'][0, 0]
            # rotate to the same size
            data = []
            num = matdata.shape[1]
            # determine the vote chosen to be edge
            if num % 2 == 0:
                vote = num / 2
            else:
                vote = np.floor(num / 2) + 1
            for i in range(0, num):
                data = matdata[0, i]['Boundaries'][0, 0]
                # count the vote
                count[data > 0] += 1
            # choose edge when count >= vote
            count[count < vote] = 0
            count[count >= vote] = 1

            if count.shape == (481, 321):  # size: H - W
                # count = cv2.resize(data, (320, 480))  # size: W - H
                count = Image.fromarray(count)
                count = np.asarray(count.rotate(90, expand=True))
            # else:
                # count = cv2.resize(count, (480, 320))
            if foldname == 'test':
                y_test.append(count)
            elif foldname == 'val':
                y_val.append(count)
            else:
                y_train.append(count)

    y_train = np.asarray(y_train).reshape((-1, 321, 481, 1)).astype('float32')
    y_test = np.asarray(y_test).reshape((-1, 321, 481, 1)).astype('float32')
    y_val = np.asarray(y_val).reshape((-1, 321, 481, 1)).astype('float32')
    np.save(DATA_PATH + 'y_train_moreThanHalf.npy', y_train)
    np.save(DATA_PATH + 'y_test_moreThanHalf.npy', y_test)
    np.save(DATA_PATH + 'y_val_moreThanHalf.npy', y_val)


def get_y_mean():
    y_train = []
    y_test = []
    y_val = []

    for foldname in tqdm(os.listdir(GROUND_PATH)):
        for img in sorted(os.listdir(os.path.join(GROUND_PATH, foldname))):
            IMG_DIR = os.path.join(GROUND_PATH, foldname, img)
            mat = loadmat(IMG_DIR)
            gt = mat['groundTruth'][0]

            n_annot = gt.shape[0]
            # determine the vote chosen to be edge
            if n_annot % 2 == 0:
                vote = n_annot / 2
            else:
                vote = np.floor(n_annot / 2) + 1

            gt = sum(gt[k]['Boundaries'][0][0] for k in range(n_annot))
            gt = gt.astype('float32')
            gt[gt < vote] = 0
            gt[gt >= vote] = 1
            if gt.shape[0] > gt.shape[1]:
                gt = gt.transpose()
            gt = cv2.resize(gt, (480, 320))

            if foldname == 'test':
                y_test.append(gt)
            elif foldname == 'val':
                y_val.append(gt)
            else:
                y_train.append(gt)

    y_train = np.asarray(y_train).reshape((-1, 320, 480, 1)).astype('float32')
    y_test = np.asarray(y_test).reshape((-1, 320, 480, 1)).astype('float32')
    y_val = np.asarray(y_val).reshape((-1, 320, 480, 1)).astype('float32')
    np.save(DATA_PATH + 'y_train_vote.npy', y_train)
    np.save(DATA_PATH + 'y_test_vote.npy', y_test)
    np.save(DATA_PATH + 'y_val_vote.npy', y_val)


# get_y_moreThanHalf()
get_y_mean()