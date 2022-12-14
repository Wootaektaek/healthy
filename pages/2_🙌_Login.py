import streamlit as st
import webbrowser
from PIL import Image, ImageOps
import numpy as np
import webbrowser
from tensorflow import keras
import tempfile
import os
import shutil
import time
import database_
import streamlit_authenticator as stauth
import datetime
from pytz import timezone



users = database_.fetch_all_users()

names=[user['name'] for user in users]
usernames=[user['key'] for user in users]
hashed_passwords=[user['password'] for user in users]

credentials = {"usernames":{}}

for un, name, pw in zip(usernames, names, hashed_passwords):
  user_dict = {"name":name,"password":pw}
  credentials["usernames"].update({un:user_dict})

authenticator = stauth.Authenticate(credentials, 'healthy', '12345', cookie_expiry_days=0)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status == False:
  st.error('아이디나 패스워드가 잘못입력되었습니다.')

if authentication_status == None:
  st.warning('아이디와 패스워드를 입력해주세요.')
 
if authentication_status:
  authenticator.logout("Logout", "sidebar")
  st.sidebar.title(f"{name}님 반갑습니다!")

  #페이지 제목 색 추가
  st.markdown("<h1 style='text-align: center; color: #9ACD32;'>사용자 맞춤 운동 추천 헬띠</h1>", unsafe_allow_html=True)

  st.markdown("#### 안녕하세요. 헬띠입니다.\
            <br>저희는 설문을 통해 사용자에게 운동을 추천하고 사진을 업로드하면 BMI를 알려줍니다.\
            <br>이를 통해 하루하루 달라지는 본인을 확인하세요.", unsafe_allow_html=True)

  sex = st.selectbox(
      '성별을 골라주세요',
      ('남성', '여성'))

  age = st.number_input('나이를 입력해주세요', min_value=1, max_value=100, value=1, step=1)

  weight = st.number_input('몸무게를 입력해주세요', min_value=1, max_value=150, value=1, step=1)

  height = st.number_input('키를 입력해주세요', min_value=1, max_value=200, value=1, step=1)

  fat = st.number_input('체지방(%)을 입력해주세요', min_value=1, max_value=50, value=1, step=1)

  goal = st.selectbox(
      '운동 목적을 골라주세요',
      ('건강향상', '체형교정', '근력향상', '경쟁'))

  drug = st.selectbox(
      '음주, 흡연 여부를 골라주세요',
      ('없음', '음주만', '흡연만', '둘다'))

  habit = st.selectbox(
      '현재 식단을 골라주세요',
      ('생식위주', '체형고려', '체중조절', '비식이요법'))

  st.write('You : ', sex, ' / ', age, ' / ',height, ' / ', goal, ' / ', drug, ' / ', habit)

# ==================================================================================
# RECOMMEND
  import pandas as pd
  from scipy import spatial

  data1=pd.read_csv('수치데이터1101.csv')

  data1.columns=['age',
                'result',
                'weigh',
                'tall',
                'fat',
                'lifting for',
                'drugs',
                'eating habits']

  data2=pd.read_csv('운동분류1110.csv')
  data2=data2.rename(columns={'Unnamed: 0':'추천'})
  data2=data2.set_index('추천')

  df1=data1[['age','weigh','tall','fat','lifting for','drugs', 'eating habits']]

  a=np.array([])

#age
  if age<18:
      a=np.append(a, 1)
      a=a.astype(int)
  elif 18<=age<=21:
      a=np.append(a, 2)
      a=a.astype(int)
  elif 22<=age<=25:
      a=np.append(a, 3)
      a=a.astype(int)
  elif 26<=age<=29:
      a=np.append(a, 4)
      a=a.astype(int)
  else:
      a=np.append(a, 5)
      a=a.astype(int)

#weigh
  if weight<56:
      a=np.append(a, 1)
      a=a.astype(int)
  elif 56<=weight<=65:
      a=np.append(a, 2)
      a=a.astype(int)
  elif 65<=weight<=73:
      a=np.append(a, 3)
      a=a.astype(int)
  elif 74<=weight<=83:
      a=np.append(a, 4)
      a=a.astype(int)
  elif 84<=weight<=92:
      a=np.append(a, 5)
      a=a.astype(int)
  elif 92<=weight<=104:
      a=np.append(a, 6)
      a=a.astype(int)
  elif 105<=weight<=119:
      a=np.append(a, 7)
      a=a.astype(int)
  else:
      a=np.append(a, 7)
      a=a.astype(int)

#tall
  if height<152:
      a=np.append(a, 1)
      a=a.astype(int)
  elif 152<=height<=166:
      a=np.append(a, 2)
      a=a.astype(int)
  elif 167<=height<=170:
      a=np.append(a, 3)
      a=a.astype(int)
  elif 171<=height<=175:
      a=np.append(a, 4)
      a=a.astype(int)
  elif 176<=height<=180:
      a=np.append(a, 5)
      a=a.astype(int)
  elif 181<=height<=185:
      a=np.append(a, 6)
      a=a.astype(int)
  elif 186<=height<=190:
      a=np.append(a, 7)
      a=a.astype(int)
  else:
      a=np.append(a, 8)
      a=a.astype(int)

#fat
  if fat<10:
      a=np.append(a, 1)
      a=a.astype(int)
  elif 11<=fat<=12:
      a=np.append(a, 2)
      a=a.astype(int)
  elif 13<=fat<=14:
      a=np.append(a, 3)
      a=a.astype(int)
  elif 15<=fat<=16:
      a=np.append(a, 4)
      a=a.astype(int)
  elif 17<=fat<=18:
      a=np.append(a, 5)
      a=a.astype(int)
  elif 19<=fat<=20:
      a=np.append(a, 6)
      a=a.astype(int)
  elif 21<=fat<=22:
      a=np.append(a, 7)
      a=a.astype(int)
  elif 23<=fat<=24:
      a=np.append(a, 8)
      a=a.astype(int)
  elif 25<=fat<=26:
      a=np.append(a, 9)
      a=a.astype(int)
  else:
      a=np.append(a, 10)
      a=a.astype(int) 

#purpose
  if goal=='건강향상':
      a=np.append(a, 1)
      a=a.astype(int)
  elif goal=='체형교정':
      a=np.append(a, 2)
      a=a.astype(int)
  elif goal=='근력향상':
      a=np.append(a, 3)
      a=a.astype(int)
  elif goal=='경쟁':
      a=np.append(a, 4)
      a=a.astype(int)

#drug
  if drug=='없음':
      a=np.append(a, 0)
      a=a.astype(int)
  elif drug=='음주만':
      a=np.append(a, 1)
      a=a.astype(int)
  elif drug=='흡연만':
      a=np.append(a, 2)
      a=a.astype(int)
  elif drug=='둘다':
      a=np.append(a, 3)
      a=a.astype(int)  

#habit
  if habit=='생식위주':
      a=np.append(a, 1)
      a=a.astype(int)
  elif habit=='체형고려':
      a=np.append(a, 2)
      a=a.astype(int)
  elif habit=='체중조절':
      a=np.append(a, 3)
      a=a.astype(int)
  elif habit=='비식이요법':
      a=np.append(a, 4)
      a=a.astype(int)

  user_similarity_scores = df1.dot(a) / (np.linalg.norm(df1, axis=1) * np.linalg.norm(a)) #사용자 입력에 대한 전체 코사인 유사도

  best_similarity=user_similarity_scores.idxmax()

  result=data1.iloc[best_similarity]['result']

  st.markdown("##### 저희는 {}님께 \'{}\'과(와) 관련된 운동을 추천드립니다.".format(name, result))

  df2=data2.loc[[result]]
  df2=df2.transpose()
  r=df2[df2[result]==1]
  recommend_list=r.index.to_list()
  st.markdown('다음은 \'{}\'과(와) 관련된 운동 목록입니다.'.format(result))

  for r in recommend_list:
              st.markdown('- ' + r)

# ==================================================================================
# FACE2BMI

  st.subheader('')
  st.subheader("얼굴 이미지로 예측해보는 BMI")

  ex_image = Image.open('leedaeho.jpeg')

  st.image(ex_image, caption='이처럼 정면을 바라본 이미지를 업로드 해주세요')

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
    test_dir='project/face2bmi/single_face/'
    mode = 'train' #'train' or 'predict'
    model_type = 'vgg16'
    model_tag = 'base'
    model_id = '{:s}_{:s}'.format(model_type, model_tag)
    model_dir = 'project/face2bmi/saved_model/model_{:s}.h5'.format(model_id)
    # model_dir = './model_{:s}.h5'.format(model_id)로 설정해보기
    bs = 8
    epochs = 2
    freeze_backbone = True # True => transfer learning; False => train from scratch
    import pandas as pd
    import os
    import json
    from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    from face2bmi_model import FacePrediction
    import glob
    alimages = os.listdir('project/face2bmi/train_aligned')
    train = pd.read_csv('project/face2bmi/train.csv')
    valid = pd.read_csv('project/face2bmi/valid.csv')
    train = train.loc[train['index'].isin(alimages)]
    valid = valid.loc[valid['index'].isin(alimages)]

    model = FacePrediction(img_dir = 'project/face2bmi/train_aligned', model_type = model_type)
    model.define_model(freeze_backbone = freeze_backbone)

    model.load_weights(model_dir)

#   if mode == 'train':
#     model_history = model.train(train, valid, bs = bs, epochs = epochs, callbacks = callbacks)
#   else:
#     model.load_weights(model_dir)

    if person_image is not None:
      save_uploaded_file('temp_file/', person_image)    
      g = os.listdir('temp_file/')[0]

    bmi_df=model.predict_df('temp_file/')
    bmi=bmi_df['bmi'].loc[0]
    dt_now = datetime.datetime.now(timezone('Asia/Seoul'))
    database_.insert_bmi(username, str(dt_now), str(bmi))
    df = database_.get_bmi(username)
    df = df.astype({'BMI':'float'})
    pd.to_datetime(df['Date'])

    st.subheader('{}님의 최근 BMI입니다.'.format(name))
    st.line_chart(df, x='Date', y='BMI')

    st.subheader('오늘의 BMI입니다.')
    y_pred=model.predict_faces('temp_file/'+g, show_img=True)

  if os.path.exists('temp_file'):
      shutil.rmtree('temp_file')
