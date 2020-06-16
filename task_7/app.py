import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


# http://localhost:5000


# app = Flask(__name__, template_folder='template') # to rename the templates folder that have the index.html file
app = Flask(__name__)



# ======================================================================================================================
# ADD ROUTES TO CREATE API
# ======================================================================================================================
@app.route('/')
def home():
    return render_template('index.html')


# EXAMPLE 1
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    form_text = [request.form.get('hate_speech_text_field')]
    # str(bytes_string, 'utf-8') # convert byte-string variable into a regular string

    return render_template('index.html')



# EXAMPLE 2
@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        year = request.form['year']
        return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(predicted_stock_price))



if __name__ == "__main__":
    app.run(debug=True)
    # app.run("localhost", "9999", debug=True)