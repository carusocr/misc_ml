import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import sequence

'''
3.1 of Applied AI with DeepLearning IBM self-pitch
Selecting number of time steps, epochs, training, and validation

- Dataset is going to be prices of Brent Crude oil.
- Time steps are 1 per day.
- Working with stateful LSTM, so training set size must be cleanly 
  divisible by batch size ( % == 0 ).
- compute training set size based on batch size

'''

batch_size = 64
epochs = 120
timesteps = 10  #timesteps = stride


def get_train_length(dataset, batch_size, test_percent):
    length = len(dataset)
    length *= 1 - test_percent
    train_length_values = []
    for x in range(int(length) - 100,int(length)):
        modulo = x % batch_size
        if (modulo == 0):   #yay
            train_length_values.append(x)
            print x
    return (max(train_length_values))

length = get_train_length(df_data_1, batch_size, 0.1)
print(length)

#Adding timesteps * 2
upper_train = length + timesteps*2
df_data_1_train = df_data_1[0:upper_train]
training_set = df_data_1_train.iloc[:,1:2].values
training_set.shape

# feature scale the data to range from 0 to 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(np.float64(training_set))
training_set_scaled.shape

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

plt.plot(training_set, color='red', label='Crudeness in the EU')
plt.title('Crudeness History')
plt.xlabel('Time (Days)')
plt.ylabel('Cost of Crudity. Crudite?')
plt.legend()
plt.show()
