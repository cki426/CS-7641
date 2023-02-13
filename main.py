#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 19:06:01 2023

@author: chenli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 12:59:46 2023

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

root = r"/Users/chenli/Desktop/Education/Gatech/2023Spring/MachineLearning/HW1"
os.chdir(root)

data = pd.read_csv("archive/heart.csv")
x_data = data.iloc[:,:-1]
y_data = data.iloc[:,-1]
corr = x_data.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

#################################### Decision Tree ###############################################

### fine tuning with CV
#. max depth
score_df = pd.DataFrame({"Max Depth": list(range(1, 8))})
kf = KFold(shuffle=True, n_splits=10)
depth_gini_train = []
depth_entr_train = []
depth_gini_test = []
depth_entr_test = []

for i in range(1,8):
    dtree_gini = tree.DecisionTreeClassifier(criterion = "gini",max_depth=i) 
    scores_gini = cross_validate(estimator=dtree_gini, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score = True)
    depth_gini_train.append(1-scores_gini["train_score"].mean())
    depth_gini_test.append(1-scores_gini["test_score"].mean())
    
    dtree_entr = tree.DecisionTreeClassifier(criterion = "entropy",max_depth=i) 
    scores_entr = cross_validate(estimator=dtree_entr, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score = True)
    depth_entr_train.append(1-scores_entr["train_score"].mean())
    depth_entr_test.append(1-scores_entr["test_score"].mean())

score_df["GINI_train"] = depth_gini_train
score_df["ENTROPY_train"] = depth_entr_train
score_df["GINI_test"] = depth_gini_test
score_df["ENTROPY_test"] = depth_entr_test
score_df["Max Depth"] = score_df["Max Depth"].astype(str)
plotdf(score_df, "Max Depth", ["GINI_train", "ENTROPY_train", "GINI_test", "ENTROPY_test"], "Max Depth", "Error Rate")



### learning curve
dtree_gini = tree.DecisionTreeClassifier(criterion = "gini",max_depth=8) 
dtree_entr = tree.DecisionTreeClassifier(criterion = "entropy",max_depth=8) 
train_size = [400, 800, 1200, 1600, 1800, 2000]
train_mean_gini, test_mean_gini = rundifftrainsize(dtree_gini, x_data, y_data, kf, "accuracy",
                 train_size = train_size)

train_mean_entr, test_mean_entr = rundifftrainsize(dtree_entr, x_data, y_data, kf, "accuracy",
                 train_size = train_size)

df0 = pd.DataFrame({"Training sizes": train_size})
df0["GINI_train"] = train_mean_gini
df0["GINI_test"] = test_mean_gini
df0["ENTROPY_train"] = train_mean_entr
df0["ENTROPY_test"] = test_mean_entr
plotdf(df0, "Training sizes", ["GINI_train", "ENTROPY_train", "GINI_test", "ENTROPY_test"], "Training Sizes", "Error Rate")


#################################### knn ###############################################
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

knn_df = pd.DataFrame({"Num Neighbors": list(range(1, 51))})
kf = KFold(shuffle=True, n_splits=10)
eucl_train = []
mahh_train = []
eucl_test = []
mahh_test = []

for i in range(1,51):
    knn_eucl = KNeighborsClassifier(i, metric = 'euclidean')
    scores_eucl = cross_validate(estimator=knn_eucl, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score=True)
    eucl_train.append(1 - scores_eucl["train_score"].mean())
    eucl_test.append(1 - scores_eucl["test_score"].mean())
    
    knn_mahh = KNeighborsClassifier(i, metric = 'manhattan')
    scores_mahh = cross_validate(estimator=knn_mahh, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score=True)
    mahh_train.append(1 - scores_mahh['train_score'].mean())
    mahh_test.append(1 - scores_mahh['test_score'].mean())

knn_df["Euclidean Train"] = eucl_train
knn_df["Manhattan Train"] = mahh_train
knn_df["Euclidean Test"] = eucl_test
knn_df["Manhattan Test"] = mahh_test

plotdf(knn_df, "Num Neighbors", ["Euclidean Train", "Manhattan Train", "Euclidean Test", "Manhattan Test"], "Num Neighbors", "Error Rate")


knn_eucl = KNeighborsClassifier(15, metric = 'euclidean')
knn_mahh = KNeighborsClassifier(15, metric = 'manhattan')

train_size = [400, 800, 1200, 1600, 1800, 2000]
train_mean_eucl, test_mean_eucl = rundifftrainsize(knn_eucl, x_data, y_data, kf, "accuracy",
                 train_size = train_size)

train_mean_mahh, test_mean_mahh = rundifftrainsize(knn_mahh, x_data, y_data, kf, "accuracy",
                 train_size = train_size)

df1 = pd.DataFrame({"Training sizes": train_size})
df1["Eucl_train"] = train_mean_eucl
df1["Eucl_test"] = test_mean_eucl
df1["Manh_train"] = train_mean_mahh
df1["Manh_test"] = test_mean_mahh
plotdf(df1, "Training sizes", ["Eucl_train", "Eucl_test", "Manh_train", "Manh_test"], "Training Sizes", "Error Rate")

#################################### svm ###############################################

c_list = [0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500, 1000]
df_linear = pd.DataFrame({"C": c_list})
kf = KFold(shuffle=True, n_splits=10)
linear_train = []
linear_test = []
train_size = [400, 800, 1200, 1600, 1800, 2000]

for i in c_list:
    svc_linear = make_pipeline(StandardScaler(), SVC(kernel = "linear", C = i))
    scores_linear = cross_validate(estimator=svc_linear, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score=True)
    linear_train.append(1 - scores_linear["train_score"].mean())
    linear_test.append(1 - scores_linear["test_score"].mean())
    
df_linear["Train"] = linear_train
df_linear["Test"] = linear_test
df_linear["C"] = df_linear["C"].astype(str)
plotdf(df_linear, "C", ["Train", "Test"], "C Value", "Error Rate")

svm_linear = make_pipeline(StandardScaler(), SVC(kernel = "linear", C = 0.5))
plotLC(svm_linear, x_data, y_data, kf, "accuracy", 
       train_size = train_size, title = "Learning Curve - SVM (Linear)")


gammas = [0.001, 0.1, 1, 2, 5, 10, 20]
df_gamma = pd.DataFrame()
rbf_train = []
rbf_test = []
for gamma in gammas:
    vc_rbf = make_pipeline(StandardScaler(), SVC(kernel = "rbf", C = 10, gamma = gamma))
    scores_rbf = cross_validate(estimator=svc_linear, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score=True)
    
    rbf_train.append(1 - scores_rbf["train_score"].mean())
    rbf_test.append(1 - scores_rbf["test_score"].mean())
df_gamma["Gamma"] = gammas
df_gamma["Train"] = rbf_train
df_gamma["Test"] = rbf_test
df_gamma["Gamma"]  = df_gamma["Gamma"] .astype(str)
plotdf(df_gamma, "Gamma", ["Train", "Test"], "Gamma", "Error Rate")

train_size = [200, 400, 600, 800, 1000, 1500, 1800]
svm_rbf = make_pipeline(StandardScaler(), SVC(kernel = "rbf", C = 0.5, gamma = 0.01))
plotLC(svm_rbf, x_data, y_data, kf, "accuracy", 
       train_size = train_size, title = "Learning Curve - SVM (RBF)")

#################################### boosting ###############################################
from sklearn.ensemble import AdaBoostClassifier

n_estimators = [10, 20, 50, 100, 200, 500]
depth_list = [1, 2, 4, 8]
ada_train = []
ada_test = []
#for d in depth_list:
#    print(d)
for n in n_estimators:
    print(n)
    ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 3), n_estimators = n, learning_rate = 0.01)
    scores = cross_validate(estimator=ada, X=x_train, y=y_train, cv=kf, n_jobs=4, scoring='accuracy', return_train_score=True)
    ada_train.append(1 - scores["train_score"].mean())
    ada_test.append(1 - scores["test_score"].mean())
df_ada = pd.DataFrame()
df_ada["n_estimator"] = n_estimators
df_ada["Train"] = ada_train
df_ada["Test"] = ada_test
df_ada["n_estimator"]  = df_ada["n_estimator"] .astype(str)

plotdf(df_ada, "n_estimator", ["Train", "Test"], "n_estimator", "Error Rate")

ada = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth = 3), n_estimators = 500, learning_rate = 0.01)
plotLC(ada, x_data, y_data, kf, "accuracy", 
       train_size = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800], title = "Learning Curve - AdaBoost (DT)")









