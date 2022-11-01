import streamlit as st
import webbrowser
import keras
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


