import streamlit as st
import webbrowser
from PIL import Image
st.title("사용자 맞춤 운동 추천 서비스 헬띠")

st.header("안녕하세요. 헬띠입니다. 설문을 통해 사용자에게 운동을 추천하고 사진을 업로드하면 BMI를 알려줍니다. 이를 통해 하루하루 달라지는 본인을 확인하세요.")

user_name = st.text_input("사용자의 이름을 입력하세요: ")

sex = st.selectbox(
    '성별을 골라주세요',
    ('남성', '여성'))

st.write('You selected:', sex)

user_face = st.camera_input('오늘의 당신을 알려주세요.')

