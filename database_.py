import os
from deta import Deta
import random
import pandas as pd

DETA_KEY='c0wu0bfo_dUGE1t3eeh4sg4KA8AmGZ29B2fTT4URD'
deta=Deta(DETA_KEY)

db=deta.Base('users_db')
bmi_db=deta.Base('bmi_db')

def insert_user(username, name, password):
  return db.put({'key':username, 'name':name, 'password':password})
# insert_user('lee','tae kwan', 'abc123')

def fetch_all_users():
  res=db.fetch()
  return res.items

def get_user(username):
  return db.get(username)

def update_user(username, updates):
  return db.update(updates, username)

def delete_user(username):
  return db.delete(username)

def insert_bmi(username, date, bmi):
  num = random.random()
  return bmi_db.put({'key':username+'+'+str(num), 'Date':date, 'BMI':bmi})

def get_bmi(username):
  res=bmi_db.fetch()
  df = pd.DataFrame(res.items, columns=['key', 'BMI', 'Date'])
  df = df[df['key'].str.contains(username)]
  return df
