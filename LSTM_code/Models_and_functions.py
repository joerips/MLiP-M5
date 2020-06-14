#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:21:13 2020

@author: bram
"""
import numpy as np 
import pandas as pd
from keras.models import Model
from keras.layers import Dense
from keras.layers import LSTM, Input


def read_files(dir_name):
    sample_submission = pd.read_csv(dir_name+'sample_submission.csv')
    sales_train_vali = pd.read_csv(dir_name+'sales_train_validation.csv')
    sell_prices = pd.read_csv(dir_name+'sell_prices.csv')
    calendar = pd.read_csv(dir_name+'calendar.csv')
    print('Read files')
    return sample_submission, sales_train_vali, sell_prices, calendar

#To reduce memory usage
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    print("Downcasted")
    return df

def model1(X_train):  
    inputs = Input(shape = (X_train.shape[1], X_train.shape[2]))
    model1 = LSTM(10, activation='relu', return_sequences=True)(inputs)
    model1 = LSTM(10, activation='relu')(model1)

    output = Dense(1, activation='sigmoid')(model1)
    
    model1 = Model(inputs=inputs, outputs=output)
    
    model1.compile(loss='mean_squared_error', optimizer='adam')
    
    return model1

def theirmodel(X_train):

    inputs = Input(shape = (X_train.shape[1], X_train.shape[2]))

    regressor = LSTM(50,  dropout=0.2, return_sequences=True)(inputs)
    regressor = LSTM(400,  dropout=0.2,  return_sequences=True)(regressor)
    regressor = LSTM(400, dropout=0.2)(regressor)
    
    output = Dense(30490)(regressor)
    
    model = Model(inputs = inputs, outputs = output)

    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model
