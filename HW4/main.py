#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:47:38 2023

@author: chenli
"""
import os
os.chdir(r'/Users/chenli/Desktop/Education/Gatech/2023Spring/MachineLearning/HW4')
from utility import *
import random
from datetime import datetime
import pandas as pd

REWARD = -0.04 # constant reward for non-terminal states
#MAX_ERROR = 10**(-3)

# Set up the initial environment
NUM_ACTIONS = 4
ACTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] # Down, Left, Up, Right

size_ls = [3, 4, 10, 20, 50, 100]
discount_ls = [0.1, 0.2, 0.5, 0.9, 0.99]
max_err_ls = [0.1, 0.01, 0.001, 0.0001]
result = {}
res = []

for NUM_ROW in size_ls:
    NUM_COL = NUM_ROW + 1
    for DISCOUNT in discount_ls:
        for MAX_ERROR in max_err_ls:
            U = constructU(NUM_ROW, NUM_COL)
            start = datetime.now()
            num_iter_val, err_ls, U = valueIteration(U, MAX_ERROR, NUM_ROW, NUM_COL, REWARD, DISCOUNT, NUM_ACTIONS, ACTIONS)
            time_val = datetime.now() - start
            policy_val = getOptimalPolicy(U, NUM_ROW, NUM_COL, NUM_ACTIONS, REWARD, DISCOUNT, ACTIONS)
            
            U2 = constructU(NUM_ROW, NUM_COL)
            start = datetime.now()
            policy_pol = [[random.randint(0, 3) for j in range(NUM_COL)] for i in range(NUM_ROW)]
            time_pol = datetime.now() - start
            num_iter_pol, policy_pol, U2 = policyIteration(policy_pol, U2, NUM_ROW, NUM_COL, NUM_ACTIONS, DISCOUNT, REWARD, ACTIONS, MAX_ERROR)
    
            R_val = U[NUM_ROW-1][0]
            R_pol = U2[NUM_ROW-1][0]
            
            res.append([NUM_ROW*NUM_COL, DISCOUNT, MAX_ERROR, num_iter_val, num_iter_pol, time_val, time_pol, R_val, R_pol])
            result[NUM_ROW] = {DISCOUNT: {MAX_ERROR: {"Val": (U, policy_val)}}}
            result[NUM_ROW][DISCOUNT][MAX_ERROR]["Pol"] = (U2, policy_pol)
    
res = pd.DataFrame(res)
res.columns = ["num states", "df", "error", "iter val", "iter pol", 
               "time val", "time pol", "value val", "val pol"]

res_numstates = res.loc[(res["error"] == 0.001)&(res["df"] == 0.99)]
for i in range(res_numstates.shape[0]):
    res_numstates.iloc[i, 5] = res_numstates.iloc[i, 5].microseconds
    res_numstates.iloc[i, 6] = res_numstates.iloc[i, 6].microseconds

res_df = res.loc[(res["error"] == 0.001)&(res["num states"] == 20)]
for i in range(res_df.shape[0]):
    res_df.iloc[i, 5] = res_df.iloc[i, 5].microseconds
    res_df.iloc[i, 6] = res_df.iloc[i, 6].microseconds
    
res_err = res.loc[(res["df"] == 0.99)&(res["num states"] == 20)]
for i in range(res_err.shape[0]):
    res_err.iloc[i, 5] = res_err.iloc[i, 5].microseconds
    res_err.iloc[i, 6] = res_err.iloc[i, 6].microseconds