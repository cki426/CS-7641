#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 20:16:24 2023

@author: chenli
"""

import pandas as pd
import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt

import os
os.chdir(r'/Users/chenli/Desktop/Education/Gatech/2023Spring/MachineLearning/HW4')

import mdptoolbox, mdptoolbox.example
from utility2 import *

discount = 0.9
epsilon = 0.001
states = [2,3,4,10,100,500]

result = []
for s in states:
    P, R = mdptoolbox.example.forest(S=s)
    v_engine = mdptoolbox.mdp.ValueIteration(P, R, discount, epsilon, 1000)
    v_engine.run() 
    
    p_engine = mdptoolbox.mdp.PolicyIteration(P, R, discount, None, 1000)
    p_engine.run()
    
    result.append([s, v_engine.iter, p_engine.iter, v_engine.time, p_engine.time])
    
result = pd.DataFrame(result)
result.columns = ["num states", "# iter val", "# iter pol", "time val", "time pol"]

diff = [x-y for x, y in zip(v_engine.V, p_engine.V)]
diff2 = [x-y for x, y in zip(v_engine.policy, p_engine.policy)]


result2 = []
for gamma in [0.1, 0.2, 0.5, 0.9, 0.99]:
    P, R = mdptoolbox.example.forest(S=500)
    v_engine = mdptoolbox.mdp.ValueIteration(P, R, gamma, epsilon, 1000)
    v_engine.run() 
    
    p_engine = mdptoolbox.mdp.PolicyIteration(P, R, gamma, None, 1000)
    p_engine.run()
    
    result2.append([s, v_engine.iter, p_engine.iter, v_engine.time, p_engine.time])

result2 = pd.DataFrame(result2)
    
result2.columns = ["num states", "# iter val", "# iter pol", "time val", "time pol"]

diff = [x-y for x, y in zip(v_engine.V, p_engine.V)]
diff2 = [x-y for x, y in zip(v_engine.policy, p_engine.policy)]

result3 = []
for epsilon in [0.1, 0.01, 0.001, 0.0001]:
    P, R = mdptoolbox.example.forest(S=500)
    v_engine = mdptoolbox.mdp.ValueIteration(P, R, 0.9, epsilon, 1000)
    v_engine.run() 
    
    p_engine = mdptoolbox.mdp.PolicyIteration(P, R, 0.9, None, 1000)
    p_engine.run()
    
    result3.append([s, v_engine.iter, p_engine.iter, v_engine.time, p_engine.time])

result3 = pd.DataFrame(result3)
    
result3.columns = ["num states", "# iter val", "# iter pol", "time val", "time pol"]

diff = [x-y for x, y in zip(v_engine.V, p_engine.V)]
diff2 = [x-y for x, y in zip(v_engine.policy, p_engine.policy)]


eps = [0.05,0.15,0.25,0.5,0.75,0.95]
iters = [100000]
q_df = trainQLearning(P, R, 100, discount=0.9, epsilon=eps, n_iter=iters)

evaluate_policy(P, R, v_engine.policy)

evaluate_policy(P, R, p_engine.policy)



















































