from healthy import *

from mtcnn.mtcnn import MTCNN
import cv2
import os

test_dir='./face2bmi/single_face/'
train_dir='./face'
test_processed_dir='./face2bmi/test_aligned'
train_processed_dir='./face2bmi/train_aligned'

# from tensorflow.keras.preprocessing.image import load_img
from keras.utils import load_img, img_to_array
from keras_vggface import utils
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def cut_negative_boundary(box):
    res = []
    for x in box['box']:
        if x < 0:
            x = 0
        res.append(x)
    box['box'] = res
    return box

mode = 'train' #'train' or 'predict'
model_type = 'vgg16'
model_tag = 'base'
model_id = '{:s}_{:s}'.format(model_type, model_tag)
model_dir = './face2bmi/saved_model/model_{:s}.h5'.format(model_id)
# model_dir = './model_{:s}.h5'.format(model_id)로 설정해보기
bs = 8
epochs = 2
freeze_backbone = True # True => transfer learning; False => train from scratch

import pandas as pd
import os
import json
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
from face2bmi_models import FacePrediction
import glob

alimages = os.listdir('./face2bmi/train_aligned')
train = pd.read_csv('./face2bmi/train.csv')
valid = pd.read_csv('./face2bmi/valid.csv')

train = train.loc[train['index'].isin(alimages)]
valid = valid.loc[valid['index'].isin(alimages)]

# create metrics, model dirs
Path('./face2bmi/metrics').mkdir(parents = True, exist_ok = True)
Path('./face2bmi/saved_model').mkdir(parents = True, exist_ok = True)

es = EarlyStopping(patience=3)
ckp = ModelCheckpoint(model_dir, save_best_only=True, save_weights_only=True, verbose=1)
tb = TensorBoard('./tb/%s'%(model_id))
callbacks = [es, ckp]

model = FacePrediction(img_dir = './face2bmi/train_aligned', model_type = model_type)
model.define_model(freeze_backbone = freeze_backbone)

if mode == 'train':
  model_history = model.train(train, valid, bs = bs, epochs = epochs, callbacks = callbacks)
else:
  model.load_weights(model_dir)

g = os.listdir(test_dir)
g = [file for file in g if file.endswith(".jpg")][0]
print(g)
model.predict_faces(test_dir+g, show_img=True)
