import os
from deta import Deta

DETA_KEY='c0wu0bfo_dUGE1t3eeh4sg4KA8AmGZ29B2fTT4URD'
deta=Deta(DETA_KEY)

db=deta.Base('users_db')

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