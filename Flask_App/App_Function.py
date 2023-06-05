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


def create_app():
    app = Flask(__name__)
    @app.route('/')
    @app.route('/upload', methods=['GET', 'POST'])
    #upload folder, process it and save it in the output folder
    def upload():
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)

                # Timestamp the filename
                new_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
                # Save the file
                save_location = os.path.join('Input_Files', new_filename)
                file.save(save_location)
                
                # Preprocess the file
                df1 = pd.read_csv(save_location)
                df = prep(df1)
                # Predict churn
                df = predict_churn(df,df1,model)

                #Save the processed Dataframe to a csv file 
                output_filename = f'{filename.split(".")[0]}_{str(datetime.now())}.csv'
                output_filepath = os.path.join('Output_Files', output_filename)
                df.to_csv(output_filepath, index=False)
                return redirect(url_for('download'))
        return render_template('upload.html')

    #Download the file
    @app.route('/download')
    def download():
        return render_template('download.html', files=os.listdir('Output_Files'))

    @app.route('/download/<filename>')
    def download_file(filename):
        return send_from_directory('Output_Files', filename)

    return app