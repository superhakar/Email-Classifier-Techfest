import streamlit as st
import pandas as pd
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt 
import numpy as np
import os, zipfile
import email
from sklearn.datasets import make_blobs
import model
import time
import plotly.graph_objects as go
import glob
import extract_msg as em
import base64


st.set_option('deprecation.showPyplotGlobalUse', False)

def upload():
    uploaded_file = st.file_uploader("Upload your input zip folder", type=["zip"])
    # upload_link = st.text_input("Upload From Link")
    if uploaded_file is not None:
        if(zipfile.is_zipfile(uploaded_file)):
            with zipfile.ZipFile(uploaded_file,"r") as zf:
                zf.extractall(path="Dataset")
    if(len(glob.glob('Dataset/*')) != 0):
        data,code = convert()
        model.preprocess(data,'train')
        wc = st.selectbox('Word clouds',code)
        wordc(wc,code)
        model.SVD(code)
        f = open("Code/code.txt",'w+')
        for i in range(len(code)):
            f.write("%s\r\n" % code[i])
        f.close()

def convert():
    fold = glob.glob('Dataset/*')
    code = []
    all_message = []
    for i in fold:
        code.append(i.replace("Dataset/",''))
        messages = []
        for j in glob.glob(i+'/*'):
            msg = em.Message(j)
            s = str(msg.subject) + str(msg.body)
            messages.append(s)
        all_message.append(messages)

    df = pd.DataFrame(columns=['mail','class'])
    for i in range(len(all_message)):
        new_df = pd.DataFrame(columns=['mail','class'])
        new_df['mail'] = all_message[i]
        new_df['class'] = np.zeros(len(all_message[i]))+i*np.ones(len(all_message[i]))
        df = pd.concat([df,new_df],ignore_index = True)
    return df,code

def converttest():
    test = glob.glob('Testset/*')
    test[0].replace("Testset/",'')
    testfold = glob.glob(test[0]+'/*')
    test_messages = []
    name = []
    for i in testfold:
        name.append(i)
        msg = em.Message(i)
        s = str(msg.subject) + str(msg.body)
        test_messages.append(s)

    for i in range(len(name)):
        name[i]=name[i].replace(test[0]+'/','')

    df = pd.DataFrame()
    df['mail'] = test_messages
    df['name'] = name
    return df

def wordc(wc,code):  
    df = pd.read_csv(r"CSV/Cleaned_Mails.csv", encoding ="latin-1") 
    comment_words = '' 
    stopwords = set(STOPWORDS) 
    # iterate through the csv file 
    for val in df.loc[df['class'] == code.index(wc),'mails']: 
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
            


def trainit():
    
    with st.beta_expander("Model 1 - TF-IDF Hyperparameters"):
        st.header("Model 1 - TF-IDF")
        h1 = st.slider('N-gram', 1,10,2)
        h2 = st.text_input("N-Estimators", "100")
    placeholder1 = st.empty()
    placeholder2 = st.empty()

    with st.beta_expander("Model 2 - LSTM Hyperparameters"):
        st.header("Model 2 - LSTM")
        h3 = st.slider('Embedding_dim', 16,40,20)
        h4 = st.slider('vocab_size', 1000,2000,1800)
        h5 = st.slider('max_length', 80,140,120)
        h6 = st.text_input("num_epochs", "25")
    placeholder3 = st.empty()
    placeholder4 = st.empty()

   
    if(st.button("Train")):
        with st.spinner(text='Training...'):
            cv,acc,loss = model.train(h1,int(h2),h3,h4,h5,int(h6))
            # time.sleep(5)
        placeholder2.success("Cross Validation Score: %.5f" % cv)
        placeholder3.success("Training accuracy: %.5f" % acc)
        placeholder4.success("Loss: %.5f" % loss)
        # placeholder3.success("accuracy: 99%")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}"><input type="button" value="Download {file_label}"></a>'
    return href


def test():
    uploaded_file = st.file_uploader("Upload your Test file/zip folder", type=["zip"])
    if uploaded_file is not None:
        if(zipfile.is_zipfile(uploaded_file)):
            with zipfile.ZipFile(uploaded_file,"r") as zf:
                zf.extractall(path="Testset")
        data = converttest()
        model.preprocess(data,test)
        score,xtrain2d,xtest2d,ins,out, dissimilar = model.Similarity()
        if(st.button("Test")):
            model.test(dissimilar)
            st.success("Done!!")
            st.markdown(get_binary_file_downloader_html('Result/result.csv', 'Result'), unsafe_allow_html=True)
            # st.markdown(get_binary_file_downloader_html('Test.zip', 'Result folder'), unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        gauge = {'axis': {'range': [None, 100]}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"}))
        st.plotly_chart(fig)
        plt.plot(xtrain2d[:, 0], xtrain2d[:, 1], '+',label="train")
        plt.plot(xtest2d[ins, 0], xtest2d[ins, 1], '*',label="test inside")
        plt.plot(xtest2d[out, 0], xtest2d[out, 1], 'x',label="test outside")
        plt.legend() 
        st.pyplot() 
