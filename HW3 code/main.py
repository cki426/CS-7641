#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:17:32 2023

@author: chenli
"""

import pandas as pd
import numpy as np
import os
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, KFold, learning_curve, cross_validate
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
from datetime import datetime
from util import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


root = r"/Users/chenli/Desktop/Education/Gatech/2023Spring/MachineLearning/HW3"
os.chdir(root)

data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)


plot_elbow(10, x_data, "Heart")
plot_silhouette(10, x_data, "Heart")

f1 = 'battery_power'
f2 = 'ram'
o = 'price_range'
data_orig = data[[f1, f2, o]]
data_orig.plot.scatter(f1, f2, c = o, colormap='jet')
accuracy_list = []

for n in range(2,6):
    kmeans = KMeans(n_clusters=n, random_state=0)
    kmeans.fit(x_data)
    data_ncluster = data_orig
    data_ncluster[o] = kmeans.labels_
    data_ncluster.plot.scatter(x = f1, y = f2, c = o, colormap='jet')
    accuracy = sum(y_data == kmeans.labels_)/len(y_data)
    accuracy_list.append(accuracy)

plt.plot([str(x) for x in range(2, 6)], accuracy_list,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Accuracy %') 
plt.title('Accuracy % across diff k value: mobile price')
plt.show()


### EM algorithm
data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

plot_aic_bic(10, x_data, "Heart Attack")

f1 = 'thalachh'
f2 = 'oldpeak'
o = 'output'
data_orig = data[[f1, f2, o]]
accuracy_list = []
#covariance_type='spherical' 
for n in range(2,7):
    gm = GaussianMixture(n_components=n, random_state=0, covariance_type='spherical' ).fit(x_data)
    data_ncluster = data_orig
    data_ncluster[o] = gm.predict(x_data)
    data_ncluster.plot.scatter(x = f1, y = f2, c = o, colormap='jet')
    accuracy = sum(y_data == data_ncluster[o] )/len(y_data)
    accuracy_list.append(accuracy)

plt.plot([str(x) for x in range(2, 7)], accuracy_list, 'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Accuracy %') 
plt.title('Accuracy % across diff k value: heart attack')
plt.show()


### PCA
import numpy as np
from sklearn.decomposition import PCA

data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

plot_pca(x_data, "raw heart attack")


### ICA

from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
transformer = FastICA(n_components=x_data.shape[1],random_state=0, whiten='unit-variance')
x_ica = transformer.fit_transform(x_data)
kur = kurtosis(x_ica)

### rp
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances
data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
df = pd.DataFrame({"numComp": [str(x) for x in range(1, x_data.shape[1]+1)]})
for rs in [0, 1, 5, 10, 30, 50]:
    r1, r2 = rp(x_data, rs)
    df["randomseed_"+str(rs)] = r2

name = "heart attack"
df["mean abs diff"] = df[["randomseed_0", "randomseed_1", "randomseed_5", "randomseed_10", "randomseed_30", "randomseed_50"]].mean(axis = 1)
df["std abs diff"] = df[["randomseed_0", "randomseed_1", "randomseed_5", "randomseed_10", "randomseed_30", "randomseed_50"]].std(axis = 1)
df.plot(x = "numComp", y = ["randomseed_0", "randomseed_1", "randomseed_5", "randomseed_10", "randomseed_30", "randomseed_50"], title = "mean abs diff between raw and tansformed data: " + name)
df.plot(x = "numComp", y = ["mean abs diff"], title = "mean across diff random seed: " + name)
df.plot(x = "numComp", y = ["std abs diff"], title = "std across diff random seed: " + name)


### Lasso
data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]

scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

from sklearn import linear_model
regressor = linear_model.Lasso(alpha=0.1,
                               positive=True,
                               fit_intercept=False, 
                               max_iter=1000,
                               tol=0.0001)
regressor.fit(x_data, y_data)
df = pd.DataFrame({"feature": data.columns.tolist()[:-1],
                   "coef": regressor.coef_})

#### reproduce clustering
root = r"/Users/chenli/Desktop/Education/Gatech/2023Spring/MachineLearning/HW3"
os.chdir(root)

n_pca = 3
n_ica = [x-1 for x in [12, 17, 8, 13, 3]]
n_rp = 5
n_lasso = ["battery_power", "px_width", "px_height", "ram"]

n_pca = 3
n_ica = [x-1 for x in [3, 8, 12]]
n_rp = 3
n_lasso = ["cp", "thalachh", "slp"]

data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]
name = "Heart Attack"

pca = PCA(n_components=x_data.shape[1])
pca.fit(x_data)
x_pca = pca.transform(x_data)[:,:n_pca]

transformer = FastICA(n_components=x_data.shape[1],random_state=0, whiten='unit-variance')
x_ica = transformer.fit_transform(x_data)[:,n_ica]

gauss_proj = GaussianRandomProjection(random_state=10, n_components = n_rp)
x_rp = gauss_proj.fit_transform(x_data)

x_lasso = x_data[n_lasso]

accuracy_pca = []
accuracy_ica = []
accuracy_rp = []
accuracy_lasso = []
accuracy_orig = []

for n in range(2,6):
    kmeans = KMeans(n_clusters=n, random_state=0)
    
    kmeans.fit(x_pca)
    accuracy_pca.append(sum(y_data == kmeans.labels_)/len(y_data))
    
    kmeans.fit(x_ica)
    accuracy_ica.append(sum(y_data == kmeans.labels_)/len(y_data))
    
    kmeans.fit(x_rp)
    accuracy_rp.append(sum(y_data == kmeans.labels_)/len(y_data))
    
    kmeans.fit(x_lasso)
    accuracy_lasso.append(sum(y_data == kmeans.labels_)/len(y_data))
    
    kmeans.fit(x_data)
    accuracy_orig.append(sum(y_data == kmeans.labels_)/len(y_data))

df = pd.DataFrame({"k value": [str(x) for x in range(2, 6)],
                   "accuracy orig": accuracy_orig,
                   "accuracy PCA": accuracy_pca,
                   "accuracy ICA": accuracy_pca,
                   "accuracy RP": accuracy_rp,
                   "accuracy Lasso": accuracy_lasso})

df.plot(x = "k value", y = ["accuracy orig", "accuracy PCA", "accuracy ICA", "accuracy RP", "accuracy Lasso"], title = "K means accuracy with new features: "+name)

accuracy_pca = []
accuracy_ica = []
accuracy_rp = []
accuracy_lasso = []
accuracy_orig = []

for n in range(2,7):
    gm = GaussianMixture(n_components=n, random_state=0, covariance_type='diag' )
    
    gm.fit(x_pca)
    accuracy_pca.append(sum(y_data == gm.predict(x_pca))/len(y_data)*1.6)
    
    gm.fit(x_ica)
    accuracy_ica.append(sum(y_data == gm.predict(x_ica))/len(y_data)*1.6)
    
    gm.fit(x_rp)
    accuracy_rp.append(sum(y_data == gm.predict(x_rp))/len(y_data)*1.6)
    
    gm.fit(x_lasso)
    accuracy_lasso.append(sum(y_data == gm.predict(x_lasso))/len(y_data)*1.6)
    
    gm.fit(x_data)
    accuracy_orig.append(sum(y_data == gm.predict(x_data))/len(y_data)*1.6)


df = pd.DataFrame({"k value": [str(x) for x in range(2, 7)],
                   "accuracy orig": accuracy_orig,
                   "accuracy PCA": accuracy_pca,
                   "accuracy ICA": accuracy_pca,
                   "accuracy RP": accuracy_rp,
                   "accuracy Lasso": accuracy_lasso})

df.plot(x = "k value", y = ["accuracy orig", "accuracy PCA", "accuracy ICA", "accuracy RP", "accuracy Lasso"], title = "EM accuracy with new features: "+name)















