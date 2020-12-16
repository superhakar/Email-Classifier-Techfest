import numpy as np
import pandas as pd
import re, string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt 
import streamlit as st
from sklearn.model_selection import cross_val_score
from sklearn.cluster import Birch
import os
from scipy.sparse import vstack,csr_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack
from xgboost import XGBClassifier


TOKENIZER = re.compile(f'([“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s):
  return TOKENIZER.sub(r' \1 ', s).split()


def SVD(code):
  df = pd.read_csv('CSV/Cleaned_Mails.csv')
  TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
  def tokenize(s):
    return TOKENIZER.sub(r' \1 ', s).split()
  tfidf_vectorizer = TfidfVectorizer(
        ngram_range=(1,1),
        tokenizer=tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(df['mails'])
  X = tfidf_vectorizer.transform(df['mails'])
  tsvd = TruncatedSVD(n_components=2)
  tsvd.fit(X)
  X_tsvd = tsvd.transform(X)
  for j in range(len(df['class'].unique())):
    plt.scatter(X_tsvd[df['class'] == j, 0], X_tsvd[df['class'] == j, 1], label = code[j])
  plt.legend()
  st.write("Classified plot:")
  st.pyplot()

def preprocess(data,met):
  def clean(text):   
    # removing paragraph numbers
    text = re.sub('[0-9]+.\t','',str(text))
    # removing new line characters
    text = re.sub('\r',' ',str(text))
    text = re.sub('\t',' ',str(text))
    # text = re.sub('\n',' ',str(text))
    # removing apostrophes
    text = re.sub("'s",'',str(text))
    # removing hyphens
    text = re.sub("-",' ',str(text))
    text = re.sub("— ",'',str(text))
    text = re.sub(":",'',str(text))
    # removing quotation marks
    text = re.sub('\"','',str(text))
    # removing salutations
    text = re.sub("Mr\.",'Mr',str(text))
    text = re.sub("Mrs\.",'Mrs',str(text))
    # removing any reference to outside text
    text = re.sub("[\(\[].*?[\)\]]", "", str(text))
    # removing numbers
    text = re.sub("\d+", "", str(text))
    # removing email ids
    text = re.sub("\S*@\S*\s?", "", str(text))
    text = re.sub(r'[^\w\s]', ' ', str(text)) 
    return text
  
  def get_text(Series, row_num_slicer):
    """returns a Series with text sliced from a list split from each message. Row_num_slicer
    tells function where to slice split text to find only the body of the message."""
    result = pd.Series(index=Series.index, dtype = Series.dtypes)
    for row, message in enumerate(Series):
        message_words = message.split('\n')
        del message_words[:row_num_slicer]
        result.iloc[row] = message_words
    return result

  def agg(lst):
    result = ''
    for i in lst:
        result = result + str(i)
    return result

  data['new_mails'] = get_text(data['mail'], 4)
  data['new_mails'] = data['new_mails'].apply(agg)
  data['cleaned_mails'] = data['new_mails'].apply(clean)
  train_mails = data['cleaned_mails'].tolist()
  stop_words = set(stopwords.words('english')) 
  stop_words = list(stop_words)
  stop_words = list(stop_words) + ['regards', 'thank', 'abc', 'xyz', 'test', 'sorry', 'thanks', 'to', 'from', 'sent', 'recieved']
  mails = []
  for mail in train_mails:
    sent = ''
    l_words = mail.split()
    for word in l_words:
        word1 = word.lower()
        if word1 not in stop_words:
            sent = sent + str(word1) + ' '
    mails.append(sent[:-1]) 
  if (met == 'train'):
    new_df = pd.DataFrame(columns=['mails','class'])
    new_df['mails'] = mails
    new_df['class'] = data['class']
    new_df.to_csv('CSV/Cleaned_Mails.csv')
  else:
    new_df = pd.DataFrame(columns=['mails','name'])
    new_df['mails'] = mails
    new_df['name'] = data['name']
    new_df.to_csv('CSV/Test_Cleaned_Mails.csv')

def export_new_feature_matrix(data_corpus):
  sensitive_words = ['change', 'address', 'street', 'detail', 'road', 'customer', 'support', 
  'transfer', 'scheme', 'member', 'pension', 'information', 'fund', 'retirement', 'insurance', 'corp', 'payment', 
  'update', 'centre', 'account', 'value', 'client', 'information', 'request', 'benefit', 'national', 'lump', 'sum', 'age']

  # data corpus is list of strings
  feature_matrix = np.zeros((len(data_corpus), len(sensitive_words)))
  for j in range(len(data_corpus)):
    temp_string = data_corpus[j]
    list_of_words = temp_string.split()
    for i in range(len(sensitive_words)):
      if sensitive_words[i] in list_of_words:
        feature_matrix[j][i] = 100
  return feature_matrix

def train(ngram=2,n_estimators=100,embedding_dim=20,vocab_size=1800,max_length=120,num_epochs=25):
  dataset = pd.read_csv('CSV/Cleaned_Mails.csv')
  tfidf = TfidfVectorizer(
        ngram_range=(1,ngram),
        tokenizer=tokenize,
        min_df=3,
        max_df=0.9,
        strip_accents='unicode',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    ).fit(dataset['mails'])
  X = tfidf.transform(dataset['mails']).toarray()
  y = dataset.iloc[:, -1].values
  fvs = export_new_feature_matrix(dataset['mails'])
  new_X = hstack([X, csr_matrix(fvs)]).toarray()
  pickle.dump(tfidf, open('Model/tfidf.sav', 'wb'))         
  classifier = XGBClassifier(n_estimators=n_estimators)
  classifier.fit(new_X,y)
  filename = 'Model/model.sav'
  pickle.dump(classifier, open(filename, 'wb'))
  model = XGBClassifier(n_estimators=n_estimators)
  scores = cross_val_score(model, X, y, cv=5)
  crossval = np.mean(scores)

  trunc_type='post'
  padding_type='post'
  oov_tok = "<OOV>"
  # training_size = 20000
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
  tokenizer.fit_on_texts(dataset['mails'])
  training_sequences = tokenizer.texts_to_sequences(dataset['mails'])
  training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  filename = 'Model/token.sav'
  pickle.dump(tokenizer, open(filename, 'wb'))
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences = True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(40)),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])
  scce = tf.keras.losses.SparseCategoricalCrossentropy()
  model.compile(loss=scce,optimizer='adam',metrics=['accuracy'])
  training_padded = np.array(training_padded)
  training_labels = np.array(dataset['class'])
  history = model.fit(training_padded, training_labels, epochs=num_epochs, verbose=1)
  model.save('Model/')
  return crossval,history.history['accuracy'][-1],history.history['loss'][-1]

def test(dissimilar):
  loaded_tfidf = pickle.load(open('Model/tfidf.sav', 'rb'))
  filename = 'Model/model.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  test = pd.read_csv('CSV/Test_Cleaned_Mails.csv')
  X_test1 = loaded_tfidf.transform(test['mails']).toarray()
  fvs = export_new_feature_matrix(test['mails'])
  new_X = hstack([X_test1, csr_matrix(fvs)]).toarray()
  y_pred1 = loaded_model.predict_proba(new_X)
  loaded_token = pickle.load(open('Model/token.sav', 'rb'))
  loaded_modellstm = tf.keras.models.load_model('Model/')
  max_length = 120
  trunc_type='post'
  padding_type='post'
  testing_sequences = loaded_token.texts_to_sequences(test['mails'])
  testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
  testing_padded = np.array(testing_padded)
  results = loaded_modellstm.predict(testing_padded)
  y_pred2 = np.array(results)
  f = open("Code/code.txt",'r')
  f1 = f.readlines()
  code=[]
  for i in f1:
    code.append(i.replace("\n",""))
  f.close()
  # y_pred3 = np.zeros(y_pred2.shape)
  # print(y_pred3.shape)
  # print("transfer" in test['mails'])
  # for i in range(len(test['mails'])):
  #   if("transfer" in test['mails'][i]):
  #     y_pred3[i][code.index("Transfers")]=1
  #   if("change" in test['mails'][i]):
  #     y_pred3[i][code.index("MDU")]=1
  # print(y_pred3)
  # print(y_pred2)
  # print(y_pred1)
  y_pred = y_pred1*0.9 + y_pred2*0.1
  print(y_pred)
  y_pred = np.argmax(y_pred, axis=1)
  df = pd.DataFrame()
  df['File Name'] = test['name']
  ystr = []
  for i in range(len(y_pred)):
    ystr.append(code[int(y_pred[i])])
  df['Classified Category'] = ystr
  df['Confidence'] = dissimilar
  df.to_csv('Result/result.csv', index = False)


def Similarity():
  train_df = pd.read_csv('CSV/Cleaned_Mails.csv') # change **
  test_df = pd.read_csv('CSV/Test_Cleaned_Mails.csv') # change **
  TOKENIZER = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
  def tokenize(s):
      return TOKENIZER.sub(r' \1 ', s).split()

  tfidf_vectorizer = TfidfVectorizer(
          ngram_range=(1, 1),
          tokenizer=tokenize,
          min_df=3,
          max_df=0.95,
          strip_accents='unicode',
          use_idf=True,
          smooth_idf=True,
          sublinear_tf=True,
          max_features=1000
      ).fit(pd.concat([train_df['mails'] , test_df['mails']],ignore_index = True))

  xtrain = tfidf_vectorizer.transform(train_df['mails']) 
  xtest = tfidf_vectorizer.transform(test_df['mails'])
  birch = Birch(n_clusters=None, threshold=0.9, branching_factor=50)
  birch.fit(xtrain)
  birch.partial_fit(xtest)
  set_diff = set(np.unique(birch.predict(xtest))) - set(np.unique(birch.predict(xtrain))) 
  count = 0
  out = []
  ins = []
  dissimilar = np.zeros(test_df.values.shape[0])
  for j, i in enumerate(birch.predict(xtest)):
      if i in set_diff:
          count += 1
          out.append(j)
      else:
          ins.append(j)

  sim_score = 1 - count/xtest.shape[0]
  out, ins = np.asarray(out, dtype=int), np.asarray(ins, dtype=int)
  dissimilar[ins] += 1
  tsvd = TruncatedSVD(n_components=2)
  tsvd.fit(vstack([xtrain, xtest], format='csr'))
  xtrain2d = tsvd.transform(xtrain)
  xtest2d = tsvd.transform(xtest)
  return sim_score*100,xtrain2d,xtest2d,ins,out, dissimilar
