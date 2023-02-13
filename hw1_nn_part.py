# -*- coding: utf-8 -*-
"""HW1 NN Part.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1NRyQYFHFvlrz3Tc2I1bQ8m-uqIKK6xkd
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
from datetime import datetime
from tensorflow import keras
from util import *

"""Part 1. Data"""

#os.chdir("ML_HW1")

data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

"""Part 2. Function"""

def nn_model(layer_params, input_shape):
  model = keras.models.Sequential()
  model.add(keras.layers.Flatten(input_shape = input_shape))
  for param in layer_params:
    model.add(keras.layers.Dense(param[0], activation = param[1]))
  return model

def organize_result(dic, hyper_name):
  df = []
  for key in dic.keys():
    temp = [key, 1 - np.mean(dic[key]["accuracy"][-25:]), 
            1- np.mean(dic[key]["val_accuracy"][-25:])]
    df.append(temp)
  df = pd.DataFrame(df)
  df.columns = [hyper_name, "train error", "test error"]
  return df

"""Part 3. Hyper Paramters tuning"""

params_list = [[(2048, 'sigmoid'), (4, 'softmax')],
               [(1024, 'sigmoid'), (4, 'softmax')],
               [(512, 'sigmoid'), (4, 'softmax')],
               [(256.0, 'sigmoid'),(4, 'softmax')],
               [(128.0, 'sigmoid'), (4, 'softmax')],
               [(64.0, 'sigmoid'), (4, 'softmax')]]

val_accuracy_dic = {}
for params in params_list:
  print(params)
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=32, epochs = 50, validation_data = (x_test, y_test))
  val_accuracy_dic[tuple(params)] = history.history

df_1layer = []
for key in val_accuracy_dic.keys():
  temp = [key[0][0], 1 - np.mean(val_accuracy_dic[key]["accuracy"][-10:]), 
          1- np.mean(val_accuracy_dic[key]["val_accuracy"][-10:])]
  df_1layer.append(temp)
df_1layer = pd.DataFrame(df_1layer)
df_1layer.columns = ["num neuron", "train error", "test error"]
plotdf(df_1layer, "num neuron", ["train error", "test error"], "num neuron", "Error Rate")

params_list = [[(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (256, 'sigmoid'), (4, 'softmax')],
               [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')],
               [(2048, 'sigmoid'), (1024, 'sigmoid'), (4, 'softmax')],
               [(2048, 'sigmoid'), (4, 'softmax')]]

val_accuracy_dic = {}
for params in params_list:
  print(params)
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=32, epochs = 50, validation_data = (x_test, y_test))
  val_accuracy_dic[tuple(params)] = history.history

df_multilayer = []
for key in val_accuracy_dic.keys():
  temp = [len(key)-1, 1 - np.mean(val_accuracy_dic[key]["accuracy"][-25:]), 
          1- np.mean(val_accuracy_dic[key]["val_accuracy"][-25:])]
  df_multilayer.append(temp)
df_multilayer = pd.DataFrame(df_multilayer)
df_multilayer.columns = ["num hidden layers", "train error", "test error"]
plotdf(df_multilayer, "num hidden layers", ["train error", "test error"], "num hidden layers", "Error Rate")

### learning rate
params = [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')]
val_accuracy_dic_lr = {}
lr_list = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
for lr in lr_list:
  print(lr)
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=32, epochs = 50, validation_data = (x_test, y_test))
  val_accuracy_dic_lr[lr] = history.history

df_lr = []
for key in val_accuracy_dic_lr.keys():
  temp = [str(key), 1 - np.mean(val_accuracy_dic_lr[key]["accuracy"][-25:]), 
          1- np.mean(val_accuracy_dic_lr[key]["val_accuracy"][-25:])]
  df_lr.append(temp)
df_lr = pd.DataFrame(df_lr)
df_lr.columns = ["learning rate", "train error", "test error"]
plotdf(df_lr, "learning rate", ["train error", "test error"], "learning rate", "Error Rate")

params = [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')]
val_accuracy_batch = {}
batch_sizes = {8, 16, 32, 64, 128, 256}
for batch_size in batch_sizes:
  print(batch_size)
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=batch_size, epochs = 50, validation_data = (x_test, y_test))
  val_accuracy_batch[str(batch_size)] = history.history

df_batch = organize_result(val_accuracy_batch, "batch size")
df_batch["batch size"] = df_batch["batch size"].astype(int)
df_batch = df_batch.sort_values(["batch size"])
df_batch["batch size"] = df_batch["batch size"].astype(str)
plotdf(df_batch, "batch size", ["train error", "test error"], "batch size", "Error Rate")

params = [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')]
val_accuracy_epoch = {}
epochs_list = {10, 20, 30, 50, 100}
for epoch in epochs_list:
  print(epoch)
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=16, epochs = epoch, validation_data = (x_test, y_test))
  val_accuracy_epoch[epoch] = history.history

df_epoch = organize_result(val_accuracy_epoch, "Epoch")
df_epoch = df_epoch.sort_values(["Epoch"])
df_epoch["Epoch"] = df_epoch["Epoch"].astype(str)
plotdf(df_epoch, "Epoch", ["train error", "test error"], "Epoch", "Error Rate")

params_list = [[(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')],
               [(2048, 'relu'), (1024, 'relu'), (512, 'relu'), (4, 'softmax')],
               [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'relu'), (4, 'softmax')],
               [(2048, 'sigmoid'), (1024, 'relu'), (512, 'relu'), (4, 'softmax')]]
names = ["sigmoid-sigmoid-sigmoid", "relu-relu-relu", "sigmoid-sigmoid-relu", "sigmoid-relu-relu"]
val_accuracy_act = {}
for i in range(len(params_list)):
  params = params_list[i]
  model = nn_model(params, (20,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=16, epochs = 50, validation_data = (x_test, y_test))
  val_accuracy_act[names[i]] = history.history

df_act = organize_result(val_accuracy_act, "activation fucntion")
plotdf(df_act, "activation fucntion", ["train error", "test error"], "Activation", "Error Rate")

"""Part 4. Learning Curve vs epochs"""

start = datetime.now()
params = [(1024, 'sigmoid'), (2, 'sigmoid')]               
model = nn_model(params, (13,))
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics = ["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs = 100, validation_data = (x_test, y_test))
elapse = datetime.now() - start
elapse.microseconds

error_rate_nn = pd.DataFrame({"Epoch": list(range(1, 101)),
                              "Training Error": [1 - x for x in history.history["accuracy"]],
                              "Validation Error": [1 - x for x in history.history["val_accuracy"]]})
plotdf(error_rate_nn, "Epoch", ["Training Error", "Validation Error"], "Epoch", "Error Rate")

"""Part 5. Learning Curve vs training set size"""

train_sizes = [400, 800, 1200, 1600, 1800, 2000]
train = []
val = []
for train_size in train_sizes:
  print(train_size)
  x_select = x_data[:train_size]
  y_select = y_data[:train_size]
  x_train, x_test, y_train, y_test = train_test_split(x_select, y_select, test_size=0.2)

  params = [(1024, 'sigmoid'), (2, 'sigmoid')]               
  model = nn_model(params, (13,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics = ["accuracy"])
  history = model.fit(x_train, y_train, batch_size=16, epochs = 30, validation_data = (x_test, y_test))
  train.append(1 - np.mean(history.history["accuracy"][-15:]))
  val.append(1 - np.mean(history.history["val_accuracy"][-15:]))

df_train_size = pd.DataFrame({"Training Sizes": train_sizes,
                              "Training Error": train,
                              "Validation Error": val})
df_train_size["Training Sizes"] = df_train_size["Training Sizes"].astype(str)
plotdf(df_train_size, "Training Sizes", ["Training Error", "Validation Error"], "Training Sizes", "Error Rate")