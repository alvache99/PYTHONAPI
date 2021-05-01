import time
import json


import psycopg2
import pandas as pd
from pandas import DataFrame
from pandas.io.json import json_normalize
import numpy as np
import sys
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle

DB_HOST = "pg-3e9c1636-alvachelawtiawei-d3d1.aivencloud.com"
DB_NAME = "defaultdb"
DB_USER = "avnadmin"
DB_PASS = "oazb2ykx4s5w3rc6"
DB_PORT = "11279"

con = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
cur = con.cursor()

cur.execute('SELECT "DayWeek", "CustTotal", "Pedestrian", "Promotions" FROM customer_target')
    # row  = cur.fetchall()
df = DataFrame(cur.fetchall())
df.columns = ["DayWeek","CustTotal","Pedestrian","Promotions"]
df['DayWeek'] = df['DayWeek'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],['1','2','3','4','5','6','7'])

cur.execute('SELECT "Target" FROM customer_target')
dft = DataFrame(cur.fetchall())
dft.columns = ["Target"]

train_x,valid_x,train_y,valid_y = train_test_split(df,dft,test_size=0.5,random_state=0)
    #    js = train_x.to_dict()
logr=LogisticRegression()
logr.fit(train_x,train_y)
result=logr.predict(valid_x)
        
pickle.dump(logr,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
       
