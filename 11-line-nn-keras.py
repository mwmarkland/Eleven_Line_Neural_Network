# Based on examples from here:
# https://rubikscode.net/2018/02/12/implementing-simple-neural-network-using-keras-with-python-example/
# and here:
# https://iamtrask.github.io/2015/07/12/basic-python-network/

# This seems to "work" but it seemed to be more painful to setup than I expected.
# Getting the data in the correct format and then making the input/output layers
# match in terms of shape was something I didn't think I'd need to do.

# This initial version doesn't do exactly what I want. I want to look at the URL below
# https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
# to try to get this working better as a classifier.

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

training_data = pd.DataFrame([[0,0,1],[0,1,1],[1,0,1],[1,1,1],[1,1,0],[0,1,0]])
training_answer = pd.DataFrame([0,0,1,1,1,0])
# Keras doesn't like DataFrames. Need list of arrays.
training_x = training_data.values
training_y = training_answer.values

train2_x = training_data.iloc[:,0:5].values
train2_y = training_answer.iloc[:,0:5].values
#print(train2_x)
#print(train2_y)
#print(training_x)
#print(training_y)
model = Sequential()

# input layer
model.add(Dense(3,input_dim=3,activation='sigmoid'))
# output layer
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mean_squared_error',optimizer='SGD',metrics=['accuracy'])

model.fit(train2_x,train2_y,epochs=20,batch_size=10)

test_data = pd.DataFrame([[1,0,0],[0,1,0],[1,1,0]])
test_answer = pd.DataFrame([1,0,1])

test_x = test_data.iloc[:,0:3]
test_y = test_answer.iloc[:,0:2]
scores = model.evaluate(test_data,test_answer)
print("\nAccuracy: %.2f%%" % scores[1]*100)

print(model.predict(test_x))



