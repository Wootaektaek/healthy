import streamlit as st
import webbrowser
from PIL import Image


st.title("사용자 맞춤 운동 추천 서비스 헬띠")

st.subheader("안녕하세요. 헬띠입니다. 설문을 통해 사용자에게 운동을 추천하고 사진을 업로드하면 BMI를 알려줍니다. 이를 통해 하루하루 달라지는 본인을 확인하세요.")

user_name = st.text_input("사용자의 이름을 입력하세요: ")

sex = st.selectbox(
    '성별을 골라주세요',
    ('남성', '여성'))

#st.write('You selected:', sex)

age = st.selectbox(
    '나이를 골라주세요',
    ('~18', '19~'))

#st.write('You selected:', age)

height = st.selectbox(
    '키를 골라주세요',
    ('~155', '156~'))

#st.write('You selected:', height)

goal = st.selectbox(
    '목적을 골라주세요',
    ('유산소', '무산소'))

#st.write('You selected:', goal)

drug = st.selectbox(
    '음주, 흡연 여부를 골라주세요',
    ('금주,금연', '음주,금연'))

#st.write('You selected:', drug)

st.write('You :', user_name,' / ', sex, ' / ', age, ' / ',height, ' / ', goal, ' / ', drug)

ex_image = Image.open('leedaeho.jpeg')

st.image(ex_image, caption='이처럼 정면을 바라봐 주세요')

user_face = st.camera_input('오늘의 당신을 알려주세요.')

# ==================================================================================
# 아래부터 face2bmi 이미지 출력 코드

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
# model.model.summary()
if mode == 'train':
  model_history = model.train(train, valid, bs = bs, epochs = epochs, callbacks = callbacks)
else:
  model.load_weights(model_dir)

g = os.listdir(test_dir)
g = [file for file in g][0]

model.predict_faces(test_dir+g, show_img=True)
