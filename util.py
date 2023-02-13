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

def rundifftrainsize(model, x, y, cv, scoring, train_size):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=x,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes = train_size
    )

    train_error = 1 - train_scores
    test_error = 1 - test_scores
    train_mean = train_error.mean(axis=1)
    test_mean = test_error.mean(axis=1)
    
    return train_mean, test_mean

def plotLC(model, x, y, cv, scoring, train_size, title):

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=x,
        y=y,
        cv=cv,
        scoring=scoring,
        train_sizes = train_size
    )

    train_error = 1 - train_scores
    test_error = 1 - test_scores
    train_mean = train_error.mean(axis=1)
    test_mean = test_error.mean(axis=1)
    
    plt.subplots(figsize=(10,8))
    plt.plot(train_sizes, train_mean, label="train")
    plt.plot(train_sizes, test_mean, label="validation")
    
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("Error Rate")
    plt.legend(loc="best")
    
    plt.show()