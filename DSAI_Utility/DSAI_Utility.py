import streamlit as st
from PIL import Image

def CSS_Property(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def All_Initialization():
    col1,col2,col3 = st.columns([1,5,1])
    img = Image.open('DSAI_Utility/DeepSphere_Logo_Final.png')
    
    with col2:
        st.image(img)
    st.markdown("<h1 style='text-align: center; color: black; font-size:25px;'>Machine Learning Application(MLOps) With Databricks powered by AWS</h1>", unsafe_allow_html=True)
    st.markdown("""
    <hr style="width:100%;height:3px;background-color:gray;border-width:10">
    """, unsafe_allow_html=True)
    choice1 =  st.sidebar.selectbox(" ",('Home','About Us'))
    choice2 =  st.sidebar.selectbox(" ",('Libraries in Scope','Prophet', 'Keras','Tensorflow','Numpy','Pandas','Streamlit'))
    choice3 =  st.sidebar.selectbox(" ",('Models Implemented','Multiple Linear Regression', 'Facebook Prophet', 'LSTM RNN-Deep Learning'))
    menu = ["Google Cloud Services in Scope","Cloud Storage", "Bigquery", "Cloud Run", "Cloud Function", "Pubsub", "Vertex AI", "Secret Manager"]
    choice = st.sidebar.selectbox(" ",menu)
    st.sidebar.write('')
    st.sidebar.write('')
    href = f'<a style="color:black;text-align: center;" href="" class="button" target="_self">Clear/Reset</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
