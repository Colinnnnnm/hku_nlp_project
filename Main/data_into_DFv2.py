# -*- coding: utf-8 -*-
"""
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
import inspect

### Objective: create 1 large dataframes (a.stock prices; b.tweet data)

class DataParser:
    def __init__(self):
        
        ## set root folder 
        self.root = ""
        
        ## set location paths
        self.data_path = os.path.join(self.root, 'data/')
        self.glove_path = os.path.join(self.root,'glove.twitter.27B.50d.txt')
        self.retrieved_path = os.path.join(self.data_path, 'tweet/raw')
        self.preprocessed_path = os.path.join(self.data_path,'tweet/preprocessed')
        self.movement_path = os.path.join(self.data_path, 'price/preprocessed')
        self.rawprices_path = os.path.join(self.data_path, 'price/raw')
        self.vocab = os.path.join(self.root,'res/vocab.txt')
        
        ## needed for processing tweets data
        self.max_n_msgs=1000
        self.max_n_words=300 
        self.lasttweetdate='2015-12-31'
        self.max_n_days=5
        
        ### stock tickers
        materials= ['XOM', 'RDS-B', 'PTR', 'CVX', 'TOT', 'BP', 'BHP', 'SNP', 'SLB', 'BBL']
        consumer_goods= ['AAPL', 'PG', 'BUD', 'KO', 'PM', 'TM', 'PEP', 'UN', 'UL', 'MO']
        healthcare= ['JNJ', 'PFE', 'NVS', 'UNH', 'MRK', 'AMGN', 'MDT', 'ABBV', 'SNY', 'CELG']
        services= ['AMZN', 'BABA', 'WMT', 'CMCSA', 'HD', 'DIS', 'MCD', 'CHTR', 'UPS', 'PCLN']
        utilities= ['NEE', 'DUK', 'D', 'SO', 'NGG', 'AEP', 'PCG', 'EXC', 'SRE', 'PPL']
        cong= ['IEP', 'HRG', 'CODI', 'REX', 'SPLP', 'PICO', 'AGFS', 'GMRE']
        finance= ['BCH', 'BSAC', 'BRK-A', 'JPM', 'WFC', 'BAC', 'V', 'C', 'HSBC', 'MA']
        industrial_goods= ['GE', 'MMM', 'BA', 'HON', 'UTX', 'LMT', 'CAT', 'GD', 'DHR', 'ABB']
        tech= ['GOOG', 'MSFT', 'FB', 'T', 'CHL', 'ORCL', 'TSM', 'VZ', 'INTC', 'CSCO']

        ss_sectlist=[materials,consumer_goods,healthcare,services,utilities,cong,finance,industrial_goods,tech]

        df_sect2=pd.DataFrame()
        ##create dataframe
        for sectname in ss_sectlist:
            
            def retrieve_name(var):
                callers_local_vars = inspect.currentframe().f_back.f_locals.items()
                return [var_name for var_name, var_val in callers_local_vars if var_val is var]

            df_sect = pd.DataFrame(list(zip(sectname)),columns=[retrieve_name(sectname)[0]]) 
            df_sect_temp=pd.melt(df_sect,id_vars=None,var_name='sect',value_name='sscode')
            df_sect2=pd.concat([df_sect2,df_sect_temp],ignore_index=True)
        
        self.df_sect2=df_sect2
        
        ## Gather all stock symbols to a list for loopint
        self.stock_symbols=[*materials,*consumer_goods,*healthcare,*services,
                       *utilities,*cong,*finance,*industrial_goods,*tech]

        self.ss_size = len(self.stock_symbols)
        pass
    @staticmethod
    def get_tweet_DF(self,ss):
        
        ### import unaligned corpora 
        stock_tweet_path = os.path.join(str(self.preprocessed_path), ss)
        
        df_corpus=pd.DataFrame(data=None)
        
        ### the last data + 1 when tweet data are recorded in the folder
        enddate=datetime.strptime(self.lasttweetdate, '%Y-%m-%d').date()
        d_d_max = enddate - timedelta(days=1)
        d_d_min = enddate - timedelta(days=365*2)
        ### d_d_min = enddate - timedelta(days=3)
        
        d=d_d_max    
        while d >= d_d_min: 
            msg_fp = os.path.join(stock_tweet_path, d.isoformat())
            
            if os.path.exists(msg_fp):
                '''
                word_mat = np.zeros([max_n_msgs, max_n_words], dtype=np.int32)
                n_word_vec = np.zeros([max_n_msgs, ], dtype=np.int32)
                ss_index_vec = np.zeros([max_n_msgs, ], dtype=np.int32)
                '''
                words_raw=[]
                msg_id_df=[]
                msg_id = 0
                with open(msg_fp, 'r') as tweet_f:
                    for line in tweet_f:
                        msg_dict = json.loads(line)
                        text = msg_dict['text']
                        timestamp = msg_dict['created_at'][11:19]
                        tweet_timestamp_str=str(d)+str(" ")+str(timestamp)
                        if not text:
                            continue
                       
                        ### only consider first XX words 
                        words = text[:self.max_n_words]
                        
                        '''
                        ## It seesm no tweet length exceed 300 words based on a sample.
                        print(ss,'msg no.', msg_id,  on ',msg_fp, '. Length of tweet is ',len(text))
                        '''
                        if len(text) > self.max_n_words:
                             print(ss,'msg no.', msg_id, ' on ',msg_fp, 'exceed max_n_words')
                                                
                        ## save as list for a tweet message
                        words_raw.append(list(words))
                        
                        ## create index
                        msg_id_df.append('msg_id_'+str(msg_id))
                        
                        ## append list as a row
                        msg_id_indf='msg_id_'+str(msg_id)
                        corpus_row = [ss, d , msg_id_indf, list(words), str(tweet_timestamp_str)]
                                                                               
                        ## recall the list as one row in dataframe 
                        df_corpus_temp=pd.DataFrame(data=corpus_row).T 
                        df_corpus_temp.columns = ['sscode', 'Date', 'msg_id_no','content','tweet_timestamp_str']  
                        
                        df_corpus=pd.concat([df_corpus,df_corpus_temp],ignore_index=True)   
                       
                        ### next msg
                        msg_id += 1
                        
                        if msg_id == self.max_n_msgs:
                            print(ss,' corpus on',msg_fp,' exceed max no. of msg.')
                            break
             
            ### goes to the next day
            d -= timedelta(days=1)
    
        return df_corpus
    
    def get_one_stock_DF(self,ss):
        
        df_corpus = self.get_tweet_DF(self,ss)
        df_corpus.sort_values(by=['Date','msg_id_no'])
        df_corpus["Date"] = pd.to_datetime(df_corpus["Date"])
        df_corpus['concatcontent']=df_corpus['content'].str.join(' ')
             
        ### import raw stock csv files
        stock_prices_raw_path = os.path.join(str(self.rawprices_path), '{}.csv'.format(ss))
        df_ss_prices = pd.read_csv(stock_prices_raw_path)

        ## stock symbol
        df_ss_prices[['sscode']]=ss 
        df_ss_prices.sort_values(by=['Date'])
        df_ss_prices["Date"] = pd.to_datetime(df_ss_prices["Date"])
                
        ## merge stock prices with sector information
        df_ss_prices=pd.merge(df_ss_prices,self.df_sect2,how='left',on='sscode')
        df_ss_prices.sort_values(by=['Date'])
        df_ss_prices.drop_duplicates(subset=['Date'])
               
        def keepdiscard_y(data):
            movement = float(data)
            if -0.005 <= movement < 0.0055:
                return str('discard')
            else:
                return str('keep')
               
        ## %difference from adjusted close previous trading day
        df_ss_prices['matched_date'] = df_ss_prices['Date']
        df_ss_prices['lag1_adjclose'] = df_ss_prices.groupby(['sscode'])['Adj Close'].shift(1)
        df_ss_prices['lag1_tradeday_date'] = df_ss_prices.groupby(['sscode'])['Date'].shift(1)
        df_ss_prices['pct_ch_adj']=(df_ss_prices['Adj Close']-df_ss_prices['lag1_adjclose'])/df_ss_prices['lag1_adjclose']
        
        ## keep or discard the daily movement as in the original paper
        df_ss_prices['keepdiscard']=df_ss_prices.apply(lambda row: keepdiscard_y(row['pct_ch_adj']), axis=1)
        

        ## merge tweet data with price data (same day, if not same after day)        
        df_ss_tweet=pd.merge_asof(df_corpus.sort_values(by=['Date']),df_ss_prices,
                                  on=['Date'],direction='forward',by='sscode',tolerance=pd.Timedelta(days=self.max_n_days))
 
        ## create next tday for matching       
        df_ss_tweet['tweet_nexttday'] =df_ss_tweet['Date']+timedelta(days=1)
                
        ## Match the adj close _after Tweet Day (next trading day)
        df_ss_prices_nexttd=df_ss_prices[['Date','matched_date','Adj Close']].copy(deep=False)
        df_ss_prices_nexttd=df_ss_prices_nexttd.rename(columns={'Date':'tweet_nexttday', 
                                                       'matched_date': 'matched_next_tdate',
                                                        'Adj Close': 'adj_close_after_tweet'})
                 
        df_ss_tweet=pd.merge_asof(df_ss_tweet.sort_values(by=['tweet_nexttday']),
                                  df_ss_prices_nexttd.sort_values(by=['tweet_nexttday']),on='tweet_nexttday',
                                  direction='forward',tolerance=pd.Timedelta(days=self.max_n_days))
       

        ## create prev tday for matching
        df_ss_tweet['tweet_prevtday']=df_ss_tweet['Date']-timedelta(days=1)
        
        ## Match the adj close _after Tweet Day (next trading day)
        df_ss_prices_prevtd=df_ss_prices[['Date','matched_date','Adj Close']].copy(deep=False)
        df_ss_prices_prevtd=df_ss_prices_prevtd.rename(columns={'Date':'tweet_prevtday', 
                                                                'matched_date': 'matched_prev_tdate',
                                                                'Adj Close': 'adj_close_bef_tweet'})

        df_ss_tweet=pd.merge_asof(df_ss_tweet.sort_values(by=['tweet_prevtday']),
                                  df_ss_prices_prevtd.sort_values(by=['tweet_prevtday']),on='tweet_prevtday',
                                  direction='backward',tolerance=pd.Timedelta(days=self.max_n_days))
        
        ## sorting        
        ## df_ss_tweet.sort_values(by=['Date','msg_id_no'])
        
        return df_ss_tweet
    
    