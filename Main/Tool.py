import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
english_stopwords = stopwords.words('english')
porter = PorterStemmer()
import numpy as np
import pandas as pd
from data_into_DFv2 import DataParser
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

class ETL(DataParser):
    def __init__(self):
        super().__init__()
        self.test_date = '2015-08-01'
        self.lookup =3
    @staticmethod
    def Extract(df):
        temp =df[['Date','concatcontent']].drop_duplicates()
        text = df.groupby("Date")['concatcontent'].sum().to_frame("Text").reset_index()
        result = pd.merge(text,df[['sscode','Date','adj_close_bef_tweet','adj_close_after_tweet','pct_ch_adj']].drop_duplicates(),on=['Date'],how='left')
        result['Movement'] = result['pct_ch_adj'].apply(lambda x: int(x>0))
        return result
    @staticmethod
    def preprocess(text):
        text = text.lower().replace("url","")
        token = word_tokenize(text)
        token = [porter.stem(word) for word in token if (word not in english_stopwords) and word.isalpha()]
        return token
    def NLP_transform(self,df):
        df['Token'] = df['Text'].apply(lambda x: self.preprocess(x))
        df['Tweet'] = df['Token'].apply(lambda x: " ".join(x))
        df['BertText'] = df['Text'].apply(lambda x: x.lower().replace("url","").replace("rt",""))
        return df
    def process(self,symbol):
        return self.NLP_transform(self.Extract(self.get_one_stock_DF(symbol)))
    def BertGenerator_(self,data,validation=0):
        df = data.copy().reset_index()
        tweet_series = df['BertText']
        price_series = TimeseriesGenerator(df['adj_close_after_tweet'], df['adj_close_after_tweet'],length=self.lookup,batch_size = len(data))
        move_series = TimeseriesGenerator(df['adj_close_after_tweet'], df['Movement'],length=self.lookup,batch_size = len(data))
        if validation>0:
            validation_index = int(len(price_series[0][0])*validation)
            tweet = np.array(list(tweet_series[self.lookup:].values)[:-validation_index])
            price = price_series[0][0][:-validation_index]
            y_train = price_series[0][1][:-validation_index]
            y_train_label = move_series[0][1][:-validation_index]
            X_train = {"Tweet":tweet,"Price":price}
            
            tweet = np.array(list(tweet_series[self.lookup:].values)[-validation_index:])
            price = price_series[0][0][-validation_index:]
            y_valid = price_series[0][1][-validation_index:]
            y_valid_label = move_series[0][1][-validation_index:]
            X_valid = {"Tweet":tweet,"Price":price}
            return X_train,y_train,X_valid,y_valid,y_train_label,y_valid_label
        tweet = np.array(tweet_series[self.lookup:].values)
        price = price_series[0][0]
        y = price_series[0][1]
        y_label = move_series[0][1]
        return {"Tweet":tweet,"Price":price},y,y_label
    def BertTechGenerator(self,symbol):
        df = self.process(symbol)
        train_period, test_period = df[df['Date']<self.test_date].sort_values('Date'), df[df['Date']>=self.test_date].sort_values('Date')
        X_train,y_train,X_valid,y_valid,y_train_label,y_valid_label = self.BertGenerator_(train_period,validation=0.1)
        X_test,y_test,y_test_label = self.BertGenerator_(test_period)
        return X_train,y_train,X_valid,y_valid,y_train_label,y_valid_label,X_test,y_test,y_test_label
    

from tensorflow.keras.utils import plot_model

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras import Model
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from tensorflow.keras.layers import Dense, Dropout , Concatenate,LSTM,Reshape,BatchNormalization
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score,accuracy_score 
import os
import numpy as np
import random
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
bert_encoder_dir = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1'
bert_preprocess_dir = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
bert_preprocess_layer = hub.KerasLayer(bert_preprocess_dir)
bert_encode_model = hub.KerasLayer(bert_encoder_dir, trainable=True)

class BertPlusTechModel(Model):
    def __init__(self):
        text_input = Input(shape=(), dtype=tf.string,name='Tweet')
        bert_inputs = bert_preprocess_layer(text_input)
        outputs = bert_encode_model(bert_inputs)
        sentiment =  Dense(4,activation='sigmoid')(outputs['pooled_output'])

        price_input = Input(shape=(3),name='Price')

        embedding_layer = Concatenate()([sentiment,price_input])
        net = Dense(16,activation='relu')(embedding_layer)
#         net = Dropout(0.1)(net)
#         net = Dense(16,activation='relu')(net)
#         net = Dense(16,activation='relu')(net)
        net = Dense(1,activation='linear')(net)
        super().__init__(inputs =[text_input,price_input], outputs = net)
        self.compile(loss='mse')
    def train(self,X_train,y_train,validation_data,verbose,batch_size):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Normal Pharse
        self.fit(X_train,y_train,epochs=30,validation_data=validation_data,callbacks=[],verbose=verbose,batch_size=batch_size)
        # Early Stopping Pharse
        hist = self.fit(X_train,y_train,epochs=100,validation_data=validation_data,callbacks=[callback],verbose=verbose,batch_size=batch_size)
        return hist
    
class BertPlusTechModel_Classify(Model):
    def __init__(self):
        text_input = Input(shape=(), dtype=tf.string,name='Tweet')
        bert_inputs = bert_preprocess_layer(text_input)
        outputs = bert_encode_model(bert_inputs)
        sentiment =  Dense(8,activation='sigmoid')(outputs['pooled_output'])

        price_input = Input(shape=(3),name='Price')

        embedding_layer = Concatenate()([sentiment,price_input])
        net = Dense(16,activation='relu')(embedding_layer)
#         net = Dropout(0.1)(net)
#         net = Dense(16,activation='relu')(net)
#         net = Dense(16,activation='relu')(net)
        net = Dense(1,activation='sigmoid')(net)
        super().__init__(inputs =[text_input,price_input], outputs = net)
        self.compile(loss='BinaryCrossentropy')
    def train(self,X_train,y_train,validation_data,verbose,batch_size):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        # Normal Pharse
        self.fit(X_train,y_train,epochs=10,validation_data=validation_data,callbacks=[],verbose=verbose,batch_size=batch_size)
        # Early Stopping Pharse
        hist = self.fit(X_train,y_train,epochs=100,validation_data=validation_data,callbacks=[callback],verbose=verbose,batch_size=batch_size)
        return hist