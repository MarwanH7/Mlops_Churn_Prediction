from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import metrics
from sklearn import compose as cmp
from sklearn import preprocessing as pre
from sklearn import pipeline as pipe
import pandas as pd
import numpy as np



    ## Preprocessing Function 
def prep(df1):
    #code to check if dataframe has a column is "unnamed" if yes drop it otherwise continue
    if str('Unnamed: 0') in df1.columns:
        df1.drop(columns =[ 'Unnamed: 0'], inplace = True)
    else:
        pass


    #Convert Churn to 1 and 0
    df1.Churn = df1.Churn.map({'Yes': 1, 'No': 0}).astype(int)
    #Categorical Columns
    catCols = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies", "Contract","PaperlessBilling","PaymentMethod"]
   #Numerical Columns
    numCols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    #One Hot Encoding
    df_F = pd.get_dummies(df1, columns=catCols)
    #Scale Numerical Columns
    scaler = MinMaxScaler()
    df_F[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df_F[['tenure', 'MonthlyCharges', 'TotalCharges']])
    return df_F
   


#Function that predicts churn probability from the preprocessed dataframe, 
def predict_churn(df_F,df1,model):

    #Using pickled ANN model Predict churrn then turn into a probability without predict_probaility
    # Identify X and y variables
    X = df_F.drop(['Churn','customerID'],axis=1)
    y = df_F.Churn
    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)
    df_F['Churn_Probability'] = model.predict(df_F[X_test.columns])
    df_P = df_F[['customerID', 'Churn_Probability']]
    #Merge the two dataframes on customerID
    df = pd.merge(df1, df_P, how = 'left', on = ['customerID'])

    #return merged dataframe 
    return df
