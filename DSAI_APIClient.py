import streamlit as st
import pandas as pd
from sklearn import preprocessing
import os
import requests
import numpy as np

os.environ['DATABRICKS_TOKEN'] = 'dapi8f4cfed33b682a6bc4aad9aabc90ee7d'


def read_data():
    data_frame = None
    col1, col2, col3, col4,col5 = st.columns([1,5,1,5,1])
    col6, col7, col8 = st.columns([0.8,9,0.8])
    
    col9, col10, col11, col12,col13 = st.columns([1,5,1,5,1])
    with col2:
        st.write('')
        st.write('')
        st.subheader('Upload Test Data(Revenue Forecast)')
        st.write('')
        st.write('')
        
        
    with col4:
        vAR_data = st.file_uploader("Choose a CSV file", accept_multiple_files=False,type=['csv'])
        st.write('')
    with col7:
        if vAR_data is not None:
            data_frame = pd.read_csv(vAR_data,header=0)
            st.write('')
            st.write('')
            st.write(data_frame)
    with col10:
        st.write('')
        st.write('')
        st.subheader('Select Model')
    with col12:
        vAR_model = st.selectbox('Select Model',('Select Model','DeepLearning Model','Facebook Prophet'))
        
    
    return data_frame,vAR_model

def data_preprocessing_prophet(data_frame):
    if data_frame is not None:
        data_frame.rename(columns = {'date_time':'ds'}, inplace = True)
        data_frame = data_frame[['ds','source_wind','destination_wind']]
        return data_frame


def data_preprocessing_nn(data_frame):
        
        df = None
        if data_frame is not None:
            data_frame = data_frame.drop(['date_time'], axis=1)
            
            convert_dict = {'product_type' : str,
                            'customer_type' : str,
                            'year':int,
                            'source': str,
                            'destination':str,
                            'price_type':str,
                            'flight':str,
                            'promocode':str,
                            'source_wind':float,
                            'destination_wind':float}

            data_frame = data_frame.astype(convert_dict)
            df = pd.DataFrame(data_frame[['year','source','destination','price_type','flight','promocode',
                                'product_type','customer_type', 'source_wind','destination_wind']].copy())
            df['source'] = df['source'].astype('category')
            df_source_code = dict(enumerate(df['source'].cat.categories))

            df['destination'] = df['destination'].astype('category')
            df_destination_code = dict(enumerate(df['destination'].cat.categories))
            df['product_type'] = df['product_type'].astype('category')
            df_product_code = dict(enumerate(df['product_type'].cat.categories))
            df['customer_type'] = df['customer_type'].astype('category')
            df_customer_code = dict(enumerate(df['customer_type'].cat.categories))




            df['price_type'] = df['price_type'].astype('category')
            df_price_code = dict(enumerate(df['price_type'].cat.categories))

            df['flight'] = df['flight'].astype('category')
            df_flight_code = dict(enumerate(df['flight'].cat.categories))

            df['promocode'] = df['promocode'].astype('category')
            df_promo_code = dict(enumerate(df['promocode'].cat.categories))


            cat_columns = df.select_dtypes(['category']).columns
            df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
            df = preprocessing.scale(df)
        return df



def create_tf_serving_json(data):
    if data is not None:
        return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model_nn(dataset):
  url = 'https://dbc-83ecb7c8-5bb1.cloud.databricks.com/model/Custom_keras_mlflow_model_new/2/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

def score_model_prophet(dataset):
  url = 'https://dbc-83ecb7c8-5bb1.cloud.databricks.com/model/ProphetCustomModel/2/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}'}
  data_json = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()



def Model_Result():
    score_model_val = None
    vAR_dataset,vAR_model = read_data()
    vAR_result_list = []
    print(vAR_model)
    col6, col7, col8 = st.columns([0.8,9,0.8])
    with col7:
        if vAR_model=='DeepLearning Model':
            vAR_processed_dataset = data_preprocessing_nn(vAR_dataset)

            score_model_val = score_model_nn(vAR_processed_dataset)
            for idx_val in score_model_val:
                vAR_result_list.append(idx_val[0])
            vAR_dataset['Predicted Revenue'] = vAR_result_list
            st.write('')
            st.write('')
            st.write(vAR_dataset)
            st.write('')
            vAR_dataset = vAR_dataset.to_csv().encode('utf-8')
            st.download_button(
   "Click Here to Download the Result",
   vAR_dataset,
   "ModelOutcome_DL.csv",
   "text/csv",
   key='download-csv-nn'
)
        elif vAR_model=='Facebook Prophet':
            st.write('')
            st.write('Note : For Prophet model, we are using only source and destination wind as features')
            vAR_processed_dataset = data_preprocessing_prophet(vAR_dataset)
            st.write('')
            score_model_val = score_model_prophet(vAR_processed_dataset)
            vAR_result = pd.DataFrame(score_model_val)
            st.write(vAR_result)
            st.write('')
            vAR_result = vAR_result.to_csv().encode('utf-8')
            st.download_button(
   "Click Here to Download the Result",
   vAR_result,
   "ModelOutcome_Prophet.csv",
   "text/csv",
   key='download-csv'
)
        else:
            return None
