#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import sequence


# In[2]:



import types
import pandas as pd
import ibm_boto3
from botocore.client import Config

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share your notebook.
client_4fac95ecdbce4f84809adeb4a03a6fec = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='',
    ibm_auth_endpoint="https://iam.ng.bluemix.net/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3-api.us-geo.objectstorage.service.networklayer.com')

body = client_4fac95ecdbce4f84809adeb4a03a6fec.get_object(Bucket='test-donotdelete-pr-7rku9jyo5o3x4y',Key='DCOILBRENTEU.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_1 = pd.read_csv(body)
df_data_1.head()


# In[5]:


print(df_data_1.shape)
df_data_1 = df_data_1[df_data_1.DCOILBRENTEU != "."]
print(df_data_1.shape)


# In[12]:


def get_train_length(dataset, batch_size, test_percent):
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100,int(length)):
        modulo = x % batch_size
        if (modulo == 0):   #yay
            train_length_values.append(x)
            #print(x)
    return (max(train_length_values))


# In[13]:


batch_size = 64
epochs = 120
timesteps = 10
length = get_train_length(df_data_1, batch_size, 0.1)
print(length)


# In[14]:


upper_train = length + timesteps*2
df_data_1_train = df_data_1[0:upper_train]
training_set = df_data_1_train.iloc[:,1:2].values
training_set.shape


# In[27]:


# feature scale the data to range from 0 to 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
# vid says float64 is just to avoid warnings? float32/16 works fine here too
# this course is so kludgey! WTF IBM.
training_set_scaled = sc.fit_transform(np.float32(training_set))
training_set_scaled.shape


# In[29]:


x_train = []
y_train = []

print(length)
print(timesteps)
print(length + timesteps)
for i in range(timesteps, length + timesteps): #timesteps should be 10 here
    x_train.append(training_set_scaled[i-timesteps:i,0])
    y_train.append(training_set_scaled[i:i+timesteps,0])

print(len(x_train))
print(len(y_train))

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
print(x_train.shape)
print(y_train.shape)


from keras.layers import Dense
from keras.layers import Input, LSTM
from keras.models import Model
import h5py

#initialize LSTM model with MAE loss function

inputs_1_mae = Input(batch_shape=(batch_size, timesteps, 1))
#                 vv number of nodes
lstm_1_mae = LSTM(10, stateful=True, return_sequences=True)(inputs_1_mae)
lstm_2_mae = LSTM(10, stateful=True, return_sequences=True)(lstm_1_mae)
output_1_mae = Dense(units=1)(lstm_2_mae) # 1 output unit, which is Brent crude oil price
regressor_mae = Model(inputs=inputs_1_mae, outputs=output_1_mae)
regressor_mae.compile(optimizer='adam', loss='mae')
regressor_mae.summary

