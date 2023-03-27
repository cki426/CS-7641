#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 18:56:50 2023

@author: chenli
"""
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def DecisionTree(criterion, max_depth, x_train, y_train, x_test, y_test):
    dtree = tree.DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)
    dtree.fit(x_train, y_train)
    pred = dtree.predict(x_test)
    
    return accuracy_score(y_test, pred), dtree.get_depth(), dtree

def SVM(kernel, x_train, y_train, x_test, y_test, C, degree=3.0, coef0=1.0, gamma=5.0):
    clf = make_pipeline(StandardScaler(), SVC(kernel = kernel, C = C, degree = degree, coef0 = coef0, gamma = gamma))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return accuracy_score(y_test, pred), clf

def getwallclock(model, x_train, y_train):
    start = datetime.now()
    model.fit(x_train, y_train)
    elapse = datetime.now() - start
    return elapse.microseconds

def plotdf(df, x_axis, y_axis_list, x_label, y_label):
    for y_axis in y_axis_list:
        plt.plot(x_axis, y_axis, data = df, label = y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def plot_silhouette(k, x,name):
    silhouette_avg = []
    for num_clusters in range(2, k+1):
     
        # initialise kmeans
         kmeans = KMeans(n_clusters=num_clusters)
         kmeans.fit(x)
         cluster_labels = kmeans.labels_
         silhouette_avg.append(silhouette_score(x, cluster_labels))
    plt.plot([str(x) for x in range(2, k+1)],silhouette_avg,'bx-')
    plt.xlabel('Values of K') 
    plt.ylabel('Silhouette score') 
    plt.title('Silhouette analysis For Optimal k: ' + name)
    plt.show()
     
def plot_elbow(k, x, name):
    wcss=[]
    for i in range(1,k+1):
        kmeans = KMeans(i)
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    
    number_clusters = [str(x) for x in range(1, k+1)]
    plt.plot(number_clusters,wcss)
    plt.title('The Elbow Curve: ' + name)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

def plot_aic_bic(k, x, name):
    aic, bic = [], []
    for i in range(1, k+1):
        gm = GaussianMixture(n_components=i, random_state=0).fit(x)
        aic.append(gm.aic(x))
        bic.append(gm.bic(x))
    
    df = pd.DataFrame({"nCluster": [str(x) for x in range(1, k+1)],
                       "AIC": aic,
                       "BIC": bic})
    df.plot(x = "nCluster", y = ["AIC", "BIC"], title = "AIC/BIC vs k value: "+name)
    plt.show()

def plot_pca(x, name):
    pca = PCA(n_components=x.shape[1])
    pca.fit(x)
    
    fig, ax = plt.subplots(figsize = (10, 5))
    plt.title('PCA Eigen Value and Variance Ratio for: '+name)
     
    # using the twinx() for creating another
    # axes object for secondary y-Axis
    ax2 = ax.twinx()
    ax.plot([str(x) for x in range(1, 1+x.shape[1])], pca.singular_values_, color = 'g')
    ax2.plot([str(x) for x in range(1, 1+x.shape[1])], pca.explained_variance_ratio_.cumsum(), color = 'b')
     
    # giving labels to the axises
    ax.set_xlabel('num PC', color = 'r')
    ax.set_ylabel('Eigen Value', color = 'g')
     
    # secondary y-axis label
    ax2.set_ylabel('EXplained Variance Ratio', color = 'b')
     
    # defining display layout
    plt.tight_layout()
     
    # show plot
    plt.show()

from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import euclidean_distances


def rp(x, rs):
    reduction_dim_gauss = []
    eps_arr_gauss = []
    mean_abs_diff_gauss = []
    mean_rel_diff_gauss = []
    for n in range(1, x.shape[1]+1):
    
        gauss_proj = GaussianRandomProjection(random_state=rs, n_components = n)
        X_transform = gauss_proj.fit_transform(x)
        dist_raw = euclidean_distances(x)
        dist_transform = euclidean_distances(X_transform)
        abs_diff_gauss = abs(dist_raw - dist_transform) 
        res_diff_gauss = 1 - dist_transform/dist_raw
        
        reduction_dim_gauss.append(100-X_transform.shape[1]/x.shape[1]*100)
        mean_abs_diff_gauss.append(np.mean(abs_diff_gauss.flatten()))
    
    return (reduction_dim_gauss, mean_abs_diff_gauss)












































    
    
    