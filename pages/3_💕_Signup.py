import streamlit as st
import database_


st.subheader("New Account")
name = st.text_input('이름을 입력해주세요.')
new_user = st.text_input("아이디를 입력해주세요.")
new_password = st.text_input("패스워드를 입력해주세요.",type='password')

hashed_passwords = stauth.Hasher(new_password)._hash(new_password)

if st.button("Create Account"):
  database_.insert_user(new_user, name, hashed_passwords)
  st.success("회원가입이 완료되었습니다!")
  st.info("로그인 화면에서 로그인 해주세요.")
