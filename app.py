import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import email
import fun
# from emaildata.text import Text

st.write("""
# Email Classifier

This app will classify emails to different categories!
""")

# st.sidebar.header('Hyperparameter Tuning')

funcs = {"Upload" : fun.upload,
        "EDA" : fun.eda,
        "Train Parameters" : fun.trainit,
        "Test" : fun.test}
    
def run():
    option = st.sidebar.radio("Select an option",("Upload","EDA","Train Parameters","Test"))
    funcs[option]()

if __name__ == "__main__":
    run()
    









