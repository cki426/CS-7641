#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 15:38:11 2023

@author: chenli
"""

import six
import sys
sys.modules['sklearn.externals.six'] = six

import mlrose
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import pandas as pd

### 4 peak
n = 10
pct = 0.15
lst = list(itertools.product([0, 1], repeat=n))
fitness = mlrose.FlipFlop()
result = []
lst2 = []
for sample in lst:
    result.append(fitness.evaluate(sample))
    lst2.append(fitness.evaluate(sample))
    

df = pd.DataFrame({"BinaryStr": lst,
                   "eval": result})
plt.plot(df["eval"])
plt.xlabel("ID of string")
plt.ylabel("Evaluation of fitness")

### fitness vs input size
pct = 0.2
fitness = mlrose.FlipFlop()
best_state = []
best_fitness = []
iterations = []
time_lst = []

for n in range(10, 110, 10):
    print(n)
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
    T = mlrose.GeomDecay()
    time_temp = []
    start_time = time.time()
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, random_state = 1,
                                                                     schedule = T, curve = True,
                                                                     max_attempts = 50, max_iters=10000)
    total_time=time.time() - start_time
    time_temp.append(total_time)
    
    start_time = time.time()
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, curve=True, 
                                                                                   restarts=1, random_state = 1,
                                                                                   max_attempts = 20, max_iters=10000)
    total_time=time.time() - start_time
    time_temp.append(total_time)
    
    start_time = time.time()
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, curve=True,
                                                                          max_attempts = 50, max_iters=10000, 
                                                                          mutation_prob = 0.1, random_state=1)
    total_time=time.time() - start_time
    time_temp.append(total_time)
    
    start_time = time.time()
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, curve=True,
                                                                             max_attempts = 50, max_iters=10000,
                                                                             keep_pct = 0.1, random_state=1, fast_mimic=1)
    total_time=time.time() - start_time
    time_temp.append(total_time)
    
    time_lst.append(time_temp)
    
    temp = [1] * int(pct * n + 1) + [0] * int(n - pct * n -1)
    
    best_possible_fitness = fitness.evaluate(temp)
    best_fitness.append([best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_mimic, best_possible_fitness])
    best_state.append([best_state_rhc, best_state_sa, best_state_ga, best_state_mimic])
    iterations.append([len(fitness_curve_rhc), len(fitness_curve_sa), 
                       len(fitness_curve_ga), len(fitness_curve_mimic)])
    
    
best_fitness_df = pd.DataFrame(best_fitness, columns = ["RHC", "SA", "GA", "MIMIC", "Best Achievable"])
best_fitness_df["Dimension"] = list(range(10, 110, 10))

iterations_df = pd.DataFrame(iterations, columns = ["RHC", "SA", "GA", "MIMIC"])
iterations_df["Dimension"] = list(range(10, 110, 10))

time_lst_df = pd.DataFrame(time_lst, columns = ["RHC", "SA", "GA", "MIMIC"])
time_lst_df["Dimension"] = list(range(10, 110, 10))

iterations_df.plot(x = "Dimension", y = ["RHC", "SA", "GA", "MIMIC"])
plt.ylabel("number of iterations")

time_lst_df.plot(x = "Dimension", y = ["RHC", "SA", "GA", "MIMIC"])
plt.ylabel("closck time in s")

best_fitness_df.plot(x = "Dimension", y = ["RHC", "SA", "GA", "MIMIC"])

### fitness vs t_pct
n = 30
best_state = []
best_fitness = []
iterations = []

for pct in range(1, 9):
    print(pct/10)
    fitness = mlrose.FourPeaks(t_pct = pct/10)
    problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
    T = mlrose.GeomDecay()
    
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, random_state = 1,
                                                                     schedule = T, curve = True,
                                                                     max_attempts = 100, max_iters=10000)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, curve=True, 
                                                                                   restarts=1, random_state = 1,
                                                                                   max_attempts = 200, max_iters=10000)
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, curve=True,
                                                                          max_attempts = 100, max_iters=10000, 
                                                                          mutation_prob = 0.001, random_state=1)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, curve=True,
                                                                             max_attempts = 100, max_iters=10000,
                                                                             keep_pct = 0.1, random_state=1, fast_mimic=1)
    
    temp = [1] * int(pct * n + 1) + [0] * int(n - pct * n -1)
    
    best_possible_fitness = fitness.evaluate(temp)
    best_fitness.append([best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_mimic, best_possible_fitness])
    best_state.append([best_state_rhc, best_state_sa, best_state_ga, best_state_mimic])
    iterations.append([len(fitness_curve_rhc), len(fitness_curve_sa), 
                       len(fitness_curve_ga), len(fitness_curve_mimic)])
    
    
best_fitness_df = pd.DataFrame(best_fitness, columns = ["RHC", "SA", "GA", "MIMIC", "Best Achievable"])
best_fitness_df["pct"] = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%"]

best_fitness_df.plot(x = "pct", y = ["RHC", "SA", "GA", "MIMIC"])


### fine tune
# 1. max attempt
n = 50

fitness = mlrose.FlipFlop()
T = mlrose.GeomDecay()
problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
best_fitness = []
for attempt in range(10, 160, 30):
    print(attempt)
    best_state_sa, best_fitness_sa, fitness_curve_sa = mlrose.simulated_annealing(problem, random_state = 1,
                                                                     schedule = T, curve = True,
                                                                     max_attempts = attempt, max_iters=10000)
    best_state_rhc, best_fitness_rhc, fitness_curve_rhc = mlrose.random_hill_climb(problem, curve=True, 
                                                                                   restarts=1, random_state = 1,
                                                                                   max_attempts = attempt, max_iters=10000)
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, curve=True, random_state = 1, mutation_prob = 0.001,
                                                                          max_attempts = attempt, max_iters=10000)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, curve=True, random_state = 1,
                                                                             max_attempts = attempt, max_iters=10000, fast_mimic=1)
    
    best_fitness.append([best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_mimic])

best_fitness_df = pd.DataFrame(best_fitness, columns = ["RHC", "SA", "GA", "MIMIC"])
best_fitness_df["max attempts"] = list(range(10, 160, 30))
best_fitness_df = best_fitness_df.loc[best_fitness_df["max attempts"] <= 150]

best_fitness_df.plot(x = "max attempts", y = ["RHC", "SA", "GA", "MIMIC"])

### muttion probability for ga
n = 50
problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
best_fitness = []

for s in [0.0001, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1]:
    print(s)
    best_state_ga, best_fitness_ga, fitness_curve_ga = mlrose.genetic_alg(problem, curve=True,
                                                                          max_attempts = 120, max_iters=10000,
                                                                          pop_size = 500 , mutation_prob = s)
    
    best_fitness.append(best_fitness_ga)

mut_df = pd.DataFrame({"mutation probability": ["0.01%", "0.1%", "1%", "2%", "5%", "10%", "20%", "25%", "50%", "100%"],
                       "evaluation": best_fitness})

mut_df.plot(x = "mutation probability", y = "evaluation")


### keep probability for ga
n = 50
problem = mlrose.DiscreteOpt(length = n, fitness_fn = fitness, maximize = True)
best_fitness = []

for s in [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1]:
    print(s)
    best_state_mimic, best_fitness_mimic, fitness_curve_mimic = mlrose.mimic(problem, pop_size=500, curve=True,
                                                                             max_attempts = attempt, max_iters=10000,
                                                                             keep_pct = s, random_state=1, fast_mimic=1 )
    
    best_fitness.append(best_fitness_mimic)

mut_df = pd.DataFrame({"keep probability": ["0.1%", "1%", "2%", "5%", "10%", "20%", "25%", "50%", "100%"],
                       "evaluation": best_fitness})

mut_df.plot(x = "keep probability", y = "evaluation")

















