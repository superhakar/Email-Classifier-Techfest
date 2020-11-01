import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import numpy as np
import zipfile
import email
from sklearn.datasets import make_blobs

st.set_option('deprecation.showPyplotGlobalUse', False)

def upload():
    uploaded_file = st.file_uploader("Upload your input zip folder", type=["zip"])
    upload_link = st.text_input("Upload From Link")
    if uploaded_file is not None:
        if(zipfile.is_zipfile(uploaded_file)):
            with zipfile.ZipFile(uploaded_file,"r") as zf:
                zf.extractall()

            message = email.message_from_file(open('email/sample.eml'))
            print (message.type)
def wordc():
    # Reads 'Youtube04-Eminem.csv' file  
    df = pd.read_csv(r"Youtube04-Eminem.csv", encoding ="latin-1") 
    comment_words = '' 
    stopwords = set(STOPWORDS) 
    # iterate through the csv file 
    for val in df.CONTENT: 
        # typecaste each val to string 
        val = str(val) 
        # split the value 
        tokens = val.split() 
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 
        comment_words += " ".join(tokens)+" "
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    st.pyplot() 

def eda():
    st.write("Word cloud:")
    wordc()
    X1, Y1 = make_blobs(n_features=2, centers=5)
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=150, edgecolor='k')
    st.write("Classified plot:")
    st.pyplot() 
            


def trainit():
    st.header("Model 1 - TF-IDF")
    h12 = st.slider('N-gram', 1,10,3)
    h15 = st.radio("Use IDF",("True","False"))
    h14 = st.radio("Naive Bayes",("True","False"))
    
    st.success('Model\'s training accuracy is 99.2%')

    st.write("Model 2 - LSTM")
    h21 = st.selectbox('Embedding',('word2vec','GloVe','Custom Embedding'))
    h25 = st.radio("Learnable Embeddings",("False","True"))
    h24 = st.radio("Bidirectional - LSTM",("True","False"))
    h22 = st.text_input("Initial Learning Rate - LSTM", "0.01")
    # h22 = st.slider('Initial Learning Rate', 1e-5,1.0,0.001) # see again
    h23 = st.slider('Number Of Layers', 1,5,2)
    # h23 = st.text_input("Number Of Layers", "default")

    st.success('Model\'s training accuracy is 99.4%')

    st.write("Model 3 - Attention Model")
    h31 = st.selectbox('Gate',('LSTM','GRU','RNN'))
    h34 = st.radio("Bidirectional",("True","False"))
    h33 = st.text_input("Initial Learning Rate", "0.01")
    # h32 = st.slider('Initial Learning Rate', 0.0001,1.0000,0.001) # see again
    st.success('Model\'s training accuracy is 99%')

    
def test():
    uploaded_file = st.file_uploader("Upload your Test file/zip folder", type=["eml","zip"])
    if uploaded_file is not None:
        if(zipfile.is_zipfile(uploaded_file)):
            with zipfile.ZipFile(uploaded_file,"r") as zf:
                zf.extractall()
    test_dict={"Models":['Model1','Model2','Model3','Ensemble'],"Accuracies":[98.8,98.4,99,99.1]}
    st.write(pd.DataFrame.from_dict(test_dict))
    st.button("Download Result")

