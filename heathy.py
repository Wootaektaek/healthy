import streamlit as st
import webbrowser
from PIL import Image, ImageOps
import numpy as np
import webbrowser
from tensorflow import keras
import cv2
import tempfile
import os
import shutil
import time

#웹 페이지 배경
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://get.wallhere.com/photo/men-black-monochrome-room-fitness-model-bodybuilding-muscle-arm-chest-black-and-white-monochrome-photography-physical-fitness-biceps-curl-46781.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url() 

#st.title("사용자 맞춤 운동 추천 서비스 헬띠")
#페이지 제목 색 추가
st.markdown("<h1 style='text-align: center; color: green;'>사용자 맞춤 운동 추천 헬띠</h1>", unsafe_allow_html=True)

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

st.image(ex_image, caption='이처럼 정면을 바라본 이미지를 업로드 해주세요')

user_face = st.camera_input('오늘의 당신을 알려주세요.')

# ==================================================================================
# 아래부터 face2bmi 이미지 출력 코드
st.subheader('')
st.subheader("얼굴 이미지로 예측해보는 BMI")

def save_uploaded_file(dir, file):
  if not os.path.exists(dir):
    os.makedirs(dir)
  with open(os.path.join(dir, file.name), 'wb') as f:
    f.write(file.getbuffer())
  return st.success('완료되었습니다!')

person_image=st.file_uploader(
  label = '이미지를 업로드 해주세요', type = ['jpg', 'png', 'jpeg'])

submit=st.button('누르면 예측을 시작합니다. 잠시만 기다려주세요!')

if submit:
  test_dir='./face2bmi/single_face/'
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

  if person_image is not None:
    save_uploaded_file('temp_file/', person_image)    
    g = os.listdir('temp_file/')
    g = [file for file in g][-1]

    y_pred=model.predict_faces('temp_file/'+g, show_img=True)

if os.path.exists('temp_file'):
    shutil.rmtree('temp_file')
