import time
import json

from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
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


app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'

con = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
cur = con.cursor()

@app.route('/time')
def get_current_time2():
    print('TimedCalled', file=sys.stdout)
    return {'time': time.time()}

@app.route('/timeToday')
def get_current_time():
    print('TimedCalled', file=sys.stdout)
    return {'time': time.time()}




CORS(app)

@app.route('/validation', methods=['POST'])
def validateAcc():
    params = request.get_json()
    username = params["useremail"]
    password = params["password"]
    cur.execute('SELECT * FROM users WHERE username = '+"'"+username+"' "+'AND password = '+"'"+password+"'")
    rows = cur.fetchall()
    
   

    
    username = 'NULL'
    for test in rows:
        username = test[0]

    return jsonify({'acc': username})


@app.route('/logisticRegressionTrain')
def model_training():
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
        
        
        pickle.dump(logr,open('model.pkl','wb'))
        model=pickle.load(open('model.pkl','rb'))
        result=logr.predict(valid_x)
        js = json.dumps(result.tolist())
        return jsonify(js)

@app.route('/logisticRegressionPrediction', methods=['POST'])
def model_prediction():
        params = request.get_json()
        model=pickle.load(open('model.pkl','rb'))
        d = {'DayWeek': 1, 'CustTotal': 450,'Pedestrian': 15000,'Promotion': 1}
        Day = int(params['Day'])
        CustTotal = int(params['weather'])
        Pedestrian = int(params['pedestrian'])
        Promotion = int(params['promotion'])
        # custTotal = ['CustTotal':450]
        # pedestrian = ['CustTotal':15000]
        # promotion = ['Promotion':1]
        lst = list()
        lst.append(Day)
        lst.append(CustTotal)
        lst.append(Pedestrian)
        lst.append(Promotion)
        ans = model.predict([np.array(lst,dtype='int64')])
        result=ans[0]
        js = json.dumps(result.tolist())
        return jsonify({'predValue':js})
        # ans=model.predict([np.array(lst,dtype='int64')])

@app.route('/coffeeTime', methods=['POST'])
def fetch_coffee():

        params = request.get_json()
        refDay = params['Day']
        cur.execute('SELECT coffee FROM test_cafe WHERE weekday = ' + "'" +refDay+"'")
        rows = cur.fetchall()
        new_list_coffee = []
        average = 0

        sum_coffee = 0

        for test in rows:
            coffee = test[0]
            new_list_coffee.append(coffee)

        for i in range(0, len(new_list_coffee)):
            new_list_coffee[i] = int(new_list_coffee[i])

        for i in range(0, len(new_list_coffee)):
            sum_coffee = sum_coffee + new_list_coffee[i]

        average = sum_coffee/(len(new_list_coffee))


        result = []
        days = 1
        for row in rows:
            d = dict()
            d['sales'] = row[0]
            d['Mondays'] = days
            days = days + 1
            result.append(d)
   
        length = len(result)
        yUnknown = len(result) - 30
        prevMonth = length - yUnknown

        for i in range(0, yUnknown):
            result.remove(result[1])

        for i in range(0, prevMonth):
            result[i]['Mondays'] = i + 1
        
        return jsonify({'completeAdd': average,'cookieList':result})


#COOKIES ##################
try:
    @app.route('/linearMonday', methods=['POST'])
    def fetch_all_movies():
        # var ref_object = request.data.
        params = request.get_json()
        refString = params['Reference']
        predictString = params['Predict']
        coffeeSoldString = params['RefSold']
        refDay = params['Day']
        coffeeSoldInt = int(coffeeSoldString)
        
        print('LinearMondayCalled', file=sys.stdout)

        cur.execute('SELECT ' + refString + ' FROM test_cafe WHERE weekday = ' + "'" +refDay+"'")


    #   cur.execute('SELECT ' + refString + ''' FROM test_cafe WHERE weekday = '1' ''')
        rows = cur.fetchall()
        # print(rows)
        new_list_coffee = []
        average = 0

        sum_coffee = 0

        for test in rows:
            coffee = test[0]
            new_list_coffee.append(coffee)

        for i in range(0, len(new_list_coffee)):
            new_list_coffee[i] = int(new_list_coffee[i])

        for i in range(0, len(new_list_coffee)):
            sum_coffee = sum_coffee + new_list_coffee[i]

        average = sum_coffee/(len(new_list_coffee))

        cur.execute('SELECT '+ predictString + ' FROM test_cafe WHERE weekday = ' + "'" +refDay+"'")

        rows = cur.fetchall()
        # print(rows)
        new_list_cookies = []
        average_cookies = 0

        sum_cookies = 0

        for test in rows:
            cookies = test[0]
            new_list_cookies.append(cookies)

        for i in range(0, len(new_list_cookies)):
            new_list_cookies[i] = int(new_list_cookies[i])

        for i in range(0, len(new_list_cookies)):
            sum_cookies = sum_cookies + new_list_cookies[i]

        average_cookies = sum_cookies/(len(new_list_cookies))

        coffeeAndCookieCov = []

        for i in range(0, len(new_list_coffee)):
            tempCovCoffee = new_list_coffee[i] - average
            tempCovCookies = new_list_cookies[i] - average_cookies
            tempCovBoth = tempCovCoffee * tempCovCookies
            coffeeAndCookieCov.append(tempCovBoth)

        coffeeVar = []

        for i in range(0, len(new_list_coffee)):
            tempCoffeeVar = (new_list_coffee[i] - average) ** 2
            coffeeVar.append(tempCoffeeVar)

        sumCoffeeVar = 0
        sumBothCov = 0
        for i in range(0, len(coffeeVar)):
            sumCoffeeVar = sumCoffeeVar + coffeeVar[i]

        for i in range(0, len(new_list_coffee)):
            sumBothCov = sumBothCov + coffeeAndCookieCov[i]

        beta_here = sumBothCov/sum_coffee

        alpha = average_cookies - beta_here * average

        x_pred = coffeeSoldInt  # coffee

        y_pred = alpha + x_pred * beta_here

        y_predstr = str(y_pred)
       # cur.execute('SELECT '+ predictString +''', id FROM test_cafe WHERE weekday = '1' ''')

        cur.execute('SELECT '+ predictString +', id FROM test_cafe WHERE weekday ='+" '" +refDay+"'" + ' ORDER BY id ASC' )

       # cur.execute('SELECT '+ predictString +''', id FROM test_cafe WHERE weekday = '1' ORDER BY id ASC ''')
        rows = cur.fetchall()
        result = []
        days = 1
        for row in rows:
            d = dict()
            d['sales'] = row[0]
            d['Mondays'] = days
            days = days + 1
            result.append(d)
   
        length = len(result)
        yUnknown = len(result) - 30
        prevMonth = length - yUnknown

        for i in range(0, yUnknown):
            result.remove(result[1])

        for i in range(0, prevMonth):
            result[i]['Mondays'] = i + 1
      


        return jsonify({'completeAdd': y_predstr,'cookieList':result})

except:
    print('error')
#COOKIE END ##############################################


#CAKIE
