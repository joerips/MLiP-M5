#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 18:15:32 2020

@author: bram
"""
# Preprocessing steps and model from:
# https://www.kaggle.com/bountyhunters/baseline-lstm-with-keras-0-7

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In order for the code to work, first get access to the functions from the 
# "Models_and_functions.py" file, by running that file, this worked when making
# use of the Spyder IDE. If this does not work simply copy those functions
# and paste them in this file.

#%% Read in functions

dataPath = "/home/bram/Documents/M1/Machine Learning in Practice/Second/m5-forecasting-accuracy/"
timesteps = 14
startDay = 350

sample_submission, sales_train_vali , sell_prices, calendar = read_files(dataPath)

dt = downcast_dtypes(sales_train_vali)

del sample_submission, sell_prices


#%% Preprocess dataset

#Take the transpose so that we have one day for each row, and 30490 items' sales as columns
dt = dt.T

#Remove id, item_id, dept_id, cat_id, store_id, state_id columns
dt = dt[6 + startDay:]
print(dt.head(5))


#%% Depending on which feature you want to add, run the desired cells starting with
# 'X'th feature..., also watch out not to re-initialize test_dataset everytime in case
# multiple features are desired.

#%% Fifth feature, adding the snap features to the weekdays and 
#   single_event days combination
test_dataset = pd.DataFrame()

CA_snap = calendar["snap_CA"]
TX_snap = calendar["snap_TX"]
WI_snap = calendar["snap_WI"]

CA_snap = pd.DataFrame(CA_snap)
TX_snap = pd.DataFrame(TX_snap)
WI_snap = pd.DataFrame(WI_snap)


CA_snap_test = CA_snap[1913:1941]
TX_snap_test = TX_snap[1913:1941]
WI_snap_test = WI_snap[1913:1941]

CA_snap = CA_snap[startDay:1913]
TX_snap = TX_snap[startDay:1913]
WI_snap = WI_snap[startDay:1913]


CA_snap.columns = ["snap_CA"]
TX_snap.columns = ["snap_TX"]
WI_snap.columns = ["snap_WI"]

CA_snap.index = dt.index
TX_snap.index = dt.index
WI_snap.index = dt.index

test_dataset = pd.concat([test_dataset, CA_snap_test], axis=1)
test_dataset = pd.concat([test_dataset, TX_snap_test], axis=1)
test_dataset = pd.concat([test_dataset, WI_snap_test], axis=1)

dt = pd.concat([dt, CA_snap], axis=1)
dt = pd.concat([dt, TX_snap], axis=1)
dt = pd.concat([dt, WI_snap], axis=1)

#%% Fourth feature, single events and weekdays combined

# Every day of the week is one-hot encoded.

weekdays = [day for day in calendar["weekday"]]
weekdays = set(weekdays)

test_dataset = pd.DataFrame()

while(len(weekdays)>0):
    
    print('weekday number: ',len(weekdays))
    specific_day = weekdays.pop()
    specific_days = pd.DataFrame(np.zeros((1969,1)))
    
    for x,y in calendar.iterrows():
        if(calendar["weekday"][x] == specific_day):
            specific_days[0][x] = 1
            
    specific_days_test = specific_days[1913:1941]
    specific_days  = specific_days[startDay:1913]
    
    specific_days.columns = [specific_day]
    specific_days.index = dt.index
    
    test_dataset = pd.concat([test_dataset, specific_days_test], axis=1)
    dt = pd.concat([dt, specific_days], axis=1)


# Every different type of event is one-hot encoded
events = calendar["event_name_1"]
names = [days for days in events if not pd.isnull(days)]
names_set = set(names)

names_set = {x for x in names_set if x==x}


while(len(names_set)>0):
    print('single_event number: ',len(names_set))
    single_event = names_set.pop()
    single_event_days = pd.DataFrame(np.zeros((1969,1)))

    for x,y in calendar.iterrows():
        if(calendar["event_name_1"][x] == single_event):
            single_event_days[0][x] = 1

    single_event_days_test = single_event_days[1913:1941]
    single_event_days = single_event_days[startDay:1913]

    single_event_days.columns = [single_event]
    single_event_days.index = dt.index

    test_dataset = pd.concat([test_dataset, single_event_days_test], axis=1)
    dt = pd.concat([dt, single_event_days], axis = 1)

#%%  First feature DaysBeforeEvent

print(dt.columns[-50:-1])

print(dt.columns.shape)



# Create dataframe with zeros for 1969 days in the calendar
daysBeforeEvent = pd.DataFrame(np.zeros((1969,1)))

# "1" is assigned to the days before the event_name_1. Since "event_name_2" is rare, it was not added.
for x,y in calendar.iterrows():
    if((pd.isnull(calendar["event_name_1"][x])) == False):
            daysBeforeEvent[0][x-1] = 1
            #if first day was an event this row will cause an exception because "x-1".
            #Since it is not i did not consider for now.

#"calendar" won't be used anymore.
# del calendar


#"daysBeforeEventTest" will be used as input for predicting (We will forecast the days 1913-1941)
daysBeforeEventTest = daysBeforeEvent[1913:1941]
#"daysBeforeEvent" will be used for training as a feature.
daysBeforeEvent = daysBeforeEvent[startDay:1913]

#Before concatanation with our main data "dt", indexes are made same and column name is changed to "oneDayBeforeEvent"
daysBeforeEvent.columns = ["oneDayBeforeEvent"]
daysBeforeEvent.index = dt.index


dt = pd.concat([dt, daysBeforeEvent], axis = 1)


print(dt.columns)

#%% Second feature, one-hot encode every event_type

State of store is different for every product, therefore one-hot encode it as
a row-vector, and append it to the start of the matrix


events = calendar["event_name_1"]
names = [days for days in events if not pd.isnull(days)]
names_set = set(names)

names_set = {x for x in names_set if x==x}

test_dataset = pd.DataFrame()

while(len(names_set)>0):
    print(len(names_set))
    single_event = names_set.pop()
    single_event_days = pd.DataFrame(np.zeros((1969,1)))

    for x,y in calendar.iterrows():
        if(calendar["event_name_1"][x] == single_event):
            single_event_days[0][x] = 1

    single_event_days_test = single_event_days[1913:1941]
    single_event_days = single_event_days[startDay:1913]

    single_event_days.columns = [single_event]
    single_event_days.index = dt.index

    test_dataset = pd.concat([test_dataset, single_event_days_test], axis=1)
    dt = pd.concat([dt, single_event_days], axis = 1)


print(dt.columns[-31:-1])

feature_amount = len(dt.columns)


#%% Third feature Weekdays

weekdays = [day for day in calendar["weekday"]]
weekdays = set(weekdays)

test_dataset = pd.DataFrame()

while(len(weekdays)>0):
    
    print(len(weekdays))
    specific_day = weekdays.pop()
    specific_days = pd.DataFrame(np.zeros((1969,1)))
    
    for x,y in calendar.iterrows():
        if(calendar["weekday"][x] == specific_day):
            specific_days[0][x] = 1
            
    specific_days_test = specific_days[1913:1941]
    specific_days  = specific_days[startDay:1913]
    
    specific_days.columns = [specific_day]
    specific_days.index = dt.index
    
    test_dataset = pd.concat([test_dataset, specific_days_test], axis=1)
    dt = pd.concat([dt, specific_days], axis=1)
    
#%% Normalize values between 0 and 1

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dt_scaled = sc.fit_transform(dt)

# (1549, 14, 30490)
# (1549, 30490)
X_train = []
y_train = []
for i in range(timesteps, 1913 - startDay):
    X_train.append(dt_scaled[i-timesteps:i])
    y_train.append(dt_scaled[i][0:30490])


#Convert to np array to be able to feed the LSTM model
X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train.shape)
print(y_train.shape)

#%% Initialize model

regressor = theirmodel(X_train) # Total params: 20,338,290

print(regressor.summary())

#%% Pictures

# mymodel = model1(X_train)
# print(mymodel.summary())

# tf.keras.utils.plot_model(mymodel, to_file="mymodel.png", show_shapes=True)

#%% Fitting the RNN to the Training set
epoch_no=32
batch_size_RNN=44
regressor.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size_RNN)

#%% Saving weights

name="snap_incorporated"

whole_string = "/home/bram/Documents/M1/Machine Learning in Practice/Second/"
whole_string+=name
whole_string+="_model.h5"

weight_string = "/home/bram/Documents/M1/Machine Learning in Practice/Second/"
weight_string+=name
weight_string+="_weights.h5"

regressor.save_weights(weight_string)
regressor.save(whole_string)


#%% Predicting

feature_amount = len(dt.columns)

inputs= dt[-timesteps:]
inputs = sc.transform(inputs)
print(inputs.shape) # base notebook (14, 30491)

X_test = []
X_test.append(inputs[0:timesteps])
X_test = np.array(X_test)
predictions = []

#%%

tsdata = np.array(test_dataset)

for j in range(timesteps,timesteps + 28):
    predicted_stock_price = regressor.predict(X_test[0,j - timesteps:j].reshape(1, timesteps, feature_amount))
    testInput = np.column_stack((np.array(predicted_stock_price), tsdata[j-timesteps,0]))

    for i in range(1, len(tsdata[0])):
        testInput = np.column_stack((testInput, tsdata[j-timesteps,i]))
    X_test = np.append(X_test, testInput).reshape(1,j + 1,feature_amount)
    predicted_stock_price = sc.inverse_transform(testInput)[:,0:30490]
    predictions.append(predicted_stock_price)

#%% Submission file

import time

submission = pd.DataFrame(data=np.array(predictions).reshape(28,30490))

submission = submission.T

submission = pd.concat((submission, submission), ignore_index=True)

sample_submission = pd.read_csv(dataPath + "/sample_submission.csv")

idColumn = sample_submission[["id"]]

submission[["id"]] = idColumn

cols = list(submission.columns)
cols = cols[-1:] + cols[:-1]
submission = submission[cols]

colsdeneme = ["id"] + [f"F{i}" for i in range (1,29)]

submission.columns = colsdeneme

currentDateTime = time.strftime("%d%m%Y_%H%M%S")

working_dir = "/home/bram/Documents/M1/Machine Learning in Practice/Second"

submission.to_csv(working_dir + "/"+name+"_submission.csv", index=False)

