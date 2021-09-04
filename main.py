import pandas as pd 
import numpy as np
import nltk
import streamlit as st
import json
import joblib
import altair as alt
import matplotlib.pyplot as plt
from preprocessing import text_preprocessing

st.set_page_config(layout="wide")

menu_selectbox = st.sidebar.selectbox(
    "Menu",
    ("Sentiment Analysis by Text", "Sentiment Analysis by File")
)

st.title("Sentiment Analysis (Bahasa Melayu)")

# load model
model1 = joblib.load('model/bahasa_sentiment_svm_model1.pkl')
model2 = joblib.load('model/bahasa_sentiment_svm_model2.pkl')

if menu_selectbox == 'Sentiment Analysis by Text':

    model_option = st.selectbox(
        "Model Selection",
        ('Model1', 'Model2')
    )

    text = st.text_input('Enter text to be analyzed',max_chars=150)

    if st.button('Analyze Text Sentiment'):
        
        # st.write(f'Text before preprocessing: {text}')

        # clean text
        clean_text = text_preprocessing(str(text))
        # st.write(f'Text after preprocessing: {clean_text}')

        # Predict
        def start_pred(pred_data):
            if pred_data[0] == 0:
                st.error('Negative')
            elif pred_data[0] == 1:
                st.success('Positive')

        if model_option == 'Model1':
            pred_data=model1.predict([clean_text])
            start_pred(pred_data)
        elif model_option == 'Model2':
            pred_data=model2.predict([clean_text])
            start_pred(pred_data)

else:
    start_analysis = False

    model_option = st.selectbox(
        "Model Selection",
        ('Model1', 'Model2')
    )

    upload_file = st.file_uploader("Upload file", type=["csv"])

    if upload_file is not None:
        upload_data = pd.read_csv(upload_file)
        del upload_file

        st.subheader("Initial Data")
        st.write(upload_data)
        st.write('Length of data:', len(upload_data))
        column = list(upload_data.columns)

        if len(column) > 1:
            st.warning("*Warning:* Only one column of text is allowed to be analyzed")
            column_select = st.selectbox(
                "Choose a column to be analyzed",
                (column)
            )

            if st.button('Select'):
                for col in column:
                    if col != column_select:
                        upload_data = upload_data.drop([col], axis=1) 
                
                st.subheader("Data after column selected")
                st.write(upload_data)
                st.subheader("Data after preprocessing")

                # clean text
                text_cleaning = lambda x: text_preprocessing(x)
                upload_data['cleaned_text'] = pd.DataFrame(upload_data[column[column.index(column_select)]].apply(text_cleaning))
                st.write(upload_data)

                def show_result(upload_data):
                    st.subheader("Result")
                    upload_data.loc[upload_data['predicted'] == 0, 'predicted'] = "Negative"
                    upload_data.loc[upload_data['predicted'] == 1, 'predicted'] = "Positive"
                    st.write(upload_data)
                    upload_data = upload_data.drop(['cleaned_text'],axis=1)
                    dl_data = upload_data.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV', dl_data, "sentiment_result.csv","text/csv",key='browser-data')

                # Predict
                if model_option == 'Model1':
                    pred_data=model1.predict(upload_data['cleaned_text'])
                    upload_data['predicted'] = pred_data
                    show_result(upload_data)
                elif model_option == 'Model2':
                    pred_data=model2.predict(upload_data['cleaned_text'])
                    upload_data['predicted'] = pred_data
                    show_result(upload_data)

        else:
            st.subheader("Data after preprocessing")

            # clean text
            text_cleaning = lambda x: text_preprocessing(x)
            upload_data['cleaned_text'] = pd.DataFrame(upload_data[column[0]].apply(text_cleaning))
            st.write(upload_data)

            def show_result(upload_data):
                st.subheader("Result")
                upload_data.loc[upload_data['predicted'] == 0, 'predicted'] = "Negative"
                upload_data.loc[upload_data['predicted'] == 1, 'predicted'] = "Positive"
                st.write(upload_data)
                upload_data = upload_data.drop(['cleaned_text'],axis=1)
                dl_data = upload_data.to_csv(index=False).encode('utf-8')
                st.download_button('Download CSV', dl_data, "sentiment_result.csv","text/csv",key='browser-data')

            # Predict
            if model_option == 'Model1':
                pred_data=model1.predict(upload_data['cleaned_text'])
                upload_data['predicted'] = pred_data
                show_result(upload_data)
            elif model_option == 'Model2':
                pred_data=model2.predict(upload_data['cleaned_text'])
                upload_data['predicted'] = pred_data
                show_result(upload_data)
            
    else:
        st.warning("*Note:* One file must be uploaded!")