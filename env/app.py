import time


from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

import psycopg2
import pandas as pd
from pandas import DataFrame
import numpy as np
import sys

DB_HOST = "pg-3e9c1636-alvachelawtiawei-d3d1.aivencloud.com"
DB_NAME = "defaultdb"
DB_USER = "avnadmin"
DB_PASS = "oazb2ykx4s5w3rc6"
DB_PORT = "11279"


app = Flask(__name__)

app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/time')
def get_current_time():
    print('TimedCalled', file=sys.stdout)
    return {'time': time.time()}


@app.route('/plusTest', methods=['POST'])
def plus_test():
    print(request.get_json(), file=sys.stdout)
    numbers = request.get_json()
    first_number = numbers['number1']
    second_number = numbers['number2']

    total = int(first_number) + int(second_number)

    completeString = str(total)
    
    return jsonify({'completeAdd': completeString})



CORS(app)

try:
    con = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER,
        password=DB_PASS, host=DB_HOST, port=DB_PORT
    )

    cur = con.cursor()

    @app.route('/linearMonday', methods=['POST'])
    def fetch_all_movies():
        # var ref_object = request.data.
        params = request.get_json()
        refString = params['Reference']
        predictString = params['Predict']
        coffeeSoldString = params['RefSold']
        coffeeSoldInt = int(coffeeSoldString)
        
        print('LinearMondayCalled', file=sys.stdout)

        cur.execute('SELECT ' + refString + ''' FROM test_cafe WHERE weekday = '1' ''')

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

        cur.execute('SELECT '+ predictString +''' cookies FROM test_cafe WHERE weekday = '1' ''')

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
        return jsonify({'completeAdd': y_predstr})

except:
    print('error')