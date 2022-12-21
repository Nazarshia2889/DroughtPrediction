# Standard libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout

# Get model
def getModel(X_train):
    model = Sequential()
    
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 6)))
    model.add(Dropout(0.2))

    # model.add(LSTM(units = 50, return_sequences = True))
    # model.add(Dropout(0.2))

    # model.add(LSTM(units = 50, return_sequences = True))
    # model.add(Dropout(0.2))

    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    model.add(Dense(units = 6))

    model.compile(optimizer = "adam", loss = 'mean_squared_error')

    return model

