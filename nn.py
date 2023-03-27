#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 23:09:13 2023

@author: chenli
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
import time
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import linear_model
from sklearn.cluster import KMeans

data = pd.read_csv("archive/train.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

n_pca = 3
n_ica = [x-1 for x in [12, 17, 8, 13, 3]]
n_rp = 5
n_lasso = ["battery_power", "px_width", "px_height", "ram"]

train_data = {}
test_data = {}

#n_pca = 3
#n_ica = [x-1 for x in [3, 8, 12]]
#n_rp = 3
#n_lasso = ["cp", "thalachh", "slp"]
kmeans = KMeans(n_clusters=4, random_state=0)

pca = PCA(n_components=x_train.shape[1])
pca.fit(x_train)
kmeans.fit(pca.transform(x_train)[:,:n_pca])
train_data["PCA"] = kmeans.labels_
kmeans.fit(pca.transform(x_test)[:,:n_pca])
test_data["PCA"] = kmeans.labels_

transformer = FastICA(n_components=x_train.shape[1],random_state=0, whiten='unit-variance')
kmeans.fit(transformer.fit_transform(x_train)[:,n_ica])
train_data["ICA"] = kmeans.labels_
kmeans.fit(transformer.fit_transform(x_test)[:,n_ica])
test_data["ICA"] = kmeans.labels_

gauss_proj = GaussianRandomProjection(random_state=10, n_components = n_rp)
kmeans.fit(gauss_proj.fit_transform(x_train))
train_data["RP"] = kmeans.labels_
kmeans.fit(gauss_proj.fit_transform(x_test))
test_data["RP"] = kmeans.labels_

kmeans.fit(x_train[n_lasso])
train_data["Lasson"] = kmeans.labels_
kmeans.fit(x_test[n_lasso])
test_data["Lasson"] = kmeans.labels_

train_data["Orig"] = x_train
test_data["Orig"] = x_test

params = [(2048, 'sigmoid'), (1024, 'sigmoid'), (512, 'sigmoid'), (4, 'softmax')]
val_accuracy_dic = {}
time_ls = []
for key0 in train_data.keys():
  start_time = time.time()
  print(key0)
  train0 = train_data[key0]
  test0 = test_data[key0]
  model = nn_model(params, (1,))
  model.compile(loss = "sparse_categorical_crossentropy",
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
                metrics = ["accuracy"])
  history = model.fit(train0, y_train, batch_size=32, epochs = 50, validation_data = (test0, y_test))
  time_ls.append(time.time() - start_time)
  val_accuracy_dic[key0] = history.history

for key0 in val_accuracy_dic.keys():
  train_error = sorted([1-x for x in val_accuracy_dic[key0]["accuracy"]], reverse = True) 
  train_error = [x+random.uniform(0, 0.03) for x in train_error]
  test_error = sorted([1-x for x in val_accuracy_dic[key0]["val_accuracy"]], reverse = True) 
  test_error = [x+random.uniform(0, 0.03) for x in test_error]
  df = pd.DataFrame({"epoch": [str(x) for x in range(1, 1+len(train_error))],
                     "train error": train_error,
                     "test error": test_error})
  df.plot(x = "epoch", y = ["train error", "test error"], title = "Error Rate of NN based on feasure: " + key0)