

"""   
In this file we will be using streamlit to create a web app that will allow users (In house marketing team, customer service, support specialists, developer support etc) to upload a csv file and get a csv file with the churn probability for each customer.
That way they have more understanding of the customers that are likely to churn and can take action to prevent it by offering discounts, promotions, etc.

"""
import streamlit as st
from io import StringIO 
import pandas as pd 
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow import keras
model = pickle.load(open("model.sav", "rb"))
from sklearn.preprocessing import MinMaxScaler


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    

    # To read file as string:
    string_data = stringio.read()
    

    # Can be used wherever a "file-like" object is accepted:
    df1 = pd.read_csv(uploaded_file)
    print(df1.head())
    st.write(df1)
    
    


## code body 
    ## Preprocessing
    #code to check if dataframe has a column is "unnamed" if yes drop it otherwise continue
    if 'Unnamed: 0' in df1.columns:
        df1.drop(columns =[ 'Unnamed: 0'], inplace = True)
    else:
        pass
    df1.Churn = df1.Churn.map({'Yes': 1, 'No': 0}).astype(int)
    catCols = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies", "Contract","PaperlessBilling","PaymentMethod"]
    numCols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_F = pd.get_dummies(df1, columns=catCols)
    scaler = MinMaxScaler()
    df_F[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df_F[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X = df_F.drop(['Churn','customerID'],axis=1)
    y = df_F.Churn   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

    #Using pickled ANN model Predict churrn then turn into a probability without predict_probaility
    df_F['Churn_Probability'] = model.predict(df_F[X_train.columns])
    df_P = df_F[['customerID', 'Churn_Probability']].head(10)

    #Merge the two dataframes on customerID
    df = pd.merge(df1, df_P, how = 'left', on = ['customerID'])
    
## 
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(df)
    st.write(df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )

