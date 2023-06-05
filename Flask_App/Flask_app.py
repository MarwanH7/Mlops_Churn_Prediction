"""   
In this file we will be using Flask to create a web app that will allow users (In house marketing team, customer service, support specialists, developer support etc) to upload a csv file and get a csv file with the churn probability for each customer.
That way they have more understanding of the customers that are likely to churn and can take action to prevent it by offering discounts, promotions, etc.

"""

#importing libraries
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import pickle
import pandas as pd
import os
from datetime import datetime
from werkzeug.utils import secure_filename
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import metrics
from sklearn import compose as cmp
from sklearn import preprocessing as pre
from sklearn import pipeline as pipe
import io 
from io import StringIO
import csv
import numpy as np
import Functions
from Functions import prep
from Functions import predict_churn
from datetime import datetime
from App_Function import create_app


#Load Model 
model = pickle.load(open("model.sav", "rb"))

# Allow files of a specific type
#ALLOWED_EXTENSIONS = set(['csv'])

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create Flask app
application = create_app()

# Create app function


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8000, debug=True)