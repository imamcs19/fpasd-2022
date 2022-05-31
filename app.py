from flask import Flask,render_template,flash, Response, redirect,url_for,session,logging,request,jsonify,make_response
import sqlite3
from flask_cors import CORS, cross_origin
import os
import threading
#from pyngrok import ngrok
import numpy as np
import pandas
from pandas import DataFrame
import os.path
import re
import string
import datetime
import json
from flask import send_file
from io import BytesIO

import cx_Oracle      # We are an Oracle shop, and this changes some things
import csv
# import StringIO       # allows you to store response object in memory instead of on disk
from io import StringIO
# from flask import make_response # Necessary imports, should be obvious

import flaskcode

import nltk
from autocorrect import spell
from gensim.summarization import summarize as g_sumn

from sklearn.feature_extraction.text import CountVectorizer

#os.environ["FLASK_ENV"] = "development"
#app = Flask(__name__, static_folder='static')

app = Flask(__name__, static_folder='static', template_folder='templates')

CORS(app, resources=r'/api/*')

# app.debug = False
app.secret_key = 'key_app'

# app = Flask(__name__)
app.config.from_object(flaskcode.default_config)
# app.config['FLASKCODE_RESOURCE_BASEPATH'] = '/path/to/resource/folder'
#app.config['FLASKCODE_RESOURCE_BASEPATH'] = './fpasd-2022'
app.config['FLASKCODE_RESOURCE_BASEPATH'] = '.'
app.register_blueprint(flaskcode.blueprint, url_prefix='/vs')

@app.route('/fphome')
def fphome():
    return render_template('fp_index2.html')

@app.route('/fppredict',methods=['POST'])
def fppredict():
    import numpy as np
    # import matplotlib.pyplot as plt
    import pandas as pd
    import os.path

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #url = os.path.join(BASE_DIR, "spam.csv")

    #BASE_DIR = os.getcwd()+'/parsetree-search'
    url = os.path.join(BASE_DIR, "spam.csv")
    
    
    df= pd.read_csv(url, encoding="latin-1")
    #df= pd.read_csv(url)
    #df = pd.read_csv(url)
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']

    # Extract Feature With CountVectorizer

    cv = CountVectorizer()

    X = cv.fit_transform(X) # Fit the Data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #Naive Bayes Classifier

    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    #Alternative Usage of Saved Model
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl','rb')

    # clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
	    message = request.form['message']
	    data = [message]
	    vect = cv.transform(data).toarray()
	    my_prediction = clf.predict(vect)
     
    hasil_tree_sample = """
                S
      __________|_________________________
    SUB         |        PEL             KET
     |          |         |               |
    FNOM       PRE      FPREP           FPREP
  ___|____      |     ____|_____      ____|______
 NN  CC   NN    VB   IN         NN   IN         NNP
 |   |    |     |    |          |    |           |
ibu dan  ayah pergi  ke       pasar  di       Jakarta
    """

    hasil_typo_sample = """
    ibbu
    daan
    ayyah
    pperge
    ker
    passar
    did
    jakart
    """

    #return render_template('fp_result.html',prediction = my_prediction)
    return render_template('fp_index2.html',prediction = my_prediction, tree = hasil_tree_sample, typo = hasil_typo_sample)

@app.route("/")
def index():
    return redirect(url_for("login"))
    #return "Hello Flask IoT Simulator Using Python - Statistika Kelas B! :D"
 
@app.route("/chart")
def chart():
    # BASE_DIR = os.getcwd()+'/parsetree-search'
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    path_data = os.path.join(BASE_DIR, "data/data.csv")
    
    df = pandas.read_csv(path_data)
    days = np.unique(df['day'])

    allValues = {}
    for day in days:
        df_ = df.query('day==' + str(day))
        labels = df_['hour']
        #print(labels)
        values_dewp = df_['DEWP']
        values_temp = df_['TEMP']
        values1 = [values_dewp, values_temp, labels]
        allValues[day] = values1
    return render_template('charts.html', days = days, values=allValues)

@app.route("/ptree",methods=["GET", "POST"])
def ptree():
  tree_sample = """
                S                                    
      __________|_________________________            
    SUB         |        PEL             KET         
     |          |         |               |           
    FNOM       PRE      FPREP           FPREP        
  ___|____      |     ____|_____      ____|______     
 NN  CC   NN    VB   IN         NN   IN         NNP  
 |   |    |     |    |          |    |           |    
ibu dan  ayah pergi  ke       pasar  di       Jakarta
      
      """



  return tree_sample.replace(" ","&nbsp&nbsp").replace("\n","<br>")

@app.route("/login",methods=["GET", "POST"])
def login():
  conn = connect_db()
  db = conn.cursor()
  msg = ""
  if request.method == "POST":
      mail = request.form["mail"]
      passw = request.form["passw"]

      rs = db.execute("SELECT * FROM user WHERE Mail=\'"+ mail +"\'"+" AND Password=\'"+ passw+"\'" + " LIMIT 1")

      conn.commit()

      hasil = []
      for v_login in rs:
          hasil.append(v_login)

      if hasil:
          session['name'] = v_login[3]
          return redirect(url_for("fphome"))
      else:
          msg = "Masukkan Username (Email) dan Password dgn Benar!"

  return render_template("login.html", msg = msg)

@app.route("/register", methods=["GET", "POST"])
def register():
  conn = connect_db()
  db = conn.cursor()
  msg = ""
  if request.method == "POST":
      mail = request.form['mail']
      uname = request.form['uname']
      passw = request.form['passw']

      #cek apakah ada yg sudah menggunakan email tersebut
      rs = db.execute("SELECT * FROM user WHERE Mail=\'"+ mail +"\'" + " LIMIT 1")

      hasil = []
      for v_login in rs:
          hasil.append(v_login)

      if hasil:
          #msg = "Masukkan Username (Email) Lain, karena Email trsbt sudah digunakan oleh user Lainnya"
          msg = "Email trsbt sudah digunakan oleh user Lainnya"
          conn.commit()
          db.close()
          conn.close()
      else:
          if (bool(mail and not mail.isspace()) and bool(uname and not uname.isspace()) and bool(passw and not passw.isspace())):
              cmd = "insert into user(Mail, Password,Name,Level) values('{}','{}','{}','{}')".format(mail,passw,uname,'1')
              conn.execute(cmd)
              conn.commit()
              db.close()
              conn.close()
              #return redirect(url_for("login"))
              msg = "Masukkan Username (Email) dan Password dgn Benar!"
              return render_template("login.html", msg = msg)
          else:
              msg = "Lengkapi isian data Anda!"
  return render_template("register.html", msg = msg)

def connect_db():
    # import os.path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # BASE_DIR = os.getcwd()+'/parsetree-search'
    db_path = os.path.join(BASE_DIR, "data.db")
    return sqlite3.connect(db_path)

@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html")

@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html")

@app.route('/getsuhu_tipe_satu', methods=["GET", "POST"])
def getsuhu_tipe_satu():

    if 'name' in session:
        name = session['name']
    else:
        name = 'Guest'

    from datetime import datetime
    import pytz
    Date = str(datetime.today().astimezone(pytz.timezone('Asia/Jakarta')).strftime('%d-%m-%Y %H:%M:%S'))

    conn = connect_db()
    db = conn.cursor()

    c = db.execute(""" SELECT * FROM  data_suhu_dll_api_openweathermap """)

    mydata = c.fetchall()
    for x in c.fetchall():
        name_v=x[0]
        data_v=x[1]
        break

    hasil = []
    for v_login in c:
        hasil.append(v_login)

    conn.commit()
    db.close()
    conn.close()

    return render_template("getsuhu_tipe_satu.html", header = mydata)
    
@app.route('/unduh_data_tipe_satu/', methods=["GET"])
def dw_data_tipe_satu():
    # name = request.args.get('name')
    conn = connect_db()
    db = conn.cursor()

    c = db.execute("SELECT * FROM data_suhu_dll_api_openweathermap")

    # def export(load_file_id):
    si = StringIO()
    cw = csv.writer(si)

    rows = c.fetchall()
    cw.writerow([i[0] for i in c.description])
    cw.writerows(rows)
    response = make_response(si.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=data.csv'
    response.headers["Content-type"] = "text/csv"
    
    conn.commit()
    db.close()
    conn.close()

    return response

@app.route('/logout')
def logout():
   # remove the name from the session if it is there
   session.pop('name', None)
   return redirect(url_for('index'))

@app.route("/p2021",methods=["GET", "POST"])
def p2021():
  conn = connect_db()
  db = conn.cursor()
  msg = ""

  return render_template("docs/assets/index.html", msg = msg)

# Start the Flask server in a new thread
# threading.Thread(target=app.run, kwargs={"use_reloader": False}).start()

if __name__ == "__main__":
  app.run()