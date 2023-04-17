#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 12:50:01 2023

@author: chenli
"""

def constructU(num_row, num_col):
    U = []
    U.append([0]*(num_col-1)+[1])
    U.append([0]*(num_col-1)+[-1])
    for row in range(num_row-1):
        U.append([0]*num_col)
    return U

def printEnvironment(arr,  NUM_ROW, NUM_COL, policy=False):
    res = ""
    for r in range(NUM_ROW):
        res += "|"
        for c in range(NUM_COL):
            if r == c == 1:
                val = "WALL"
            elif r <= 1 and c == NUM_COL:
                val = "+1" if r == 0 else "-1"
            else:
                if policy:
                    val = ["Down", "Left", "Up", "Right"][arr[r][c]]
                else:
                    val = str(arr[r][c])
            res += " " + val[:5].ljust(5) + " |" # format
        res += "\n"
    print(res)

# Get the utility of the state reached by performing the given action from the given state
def getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, action):
    dr, dc = ACTIONS[action]
    newR, newC = r+dr, c+dc
    if newR < 0 or newC < 0 or newR >= NUM_ROW or newC >= NUM_COL or (newR == newC == 1): # collide with the boundary or the wall
        return U[r][c]
    else:
        return U[newR][newC]

# Calculate the utility of a state given an action
def calculateU(U, r, c, action, REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS):
    u = REWARD
    u += 0.1 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, (action-1)%4)
    u += 0.8 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, action)
    u += 0.1 * DISCOUNT * getU(U, r, c, NUM_ROW, NUM_COL, ACTIONS, (action+1)%4)
    return u

def valueIteration(U, MAX_ERROR, NUM_ROW, NUM_COL, REWARD, DISCOUNT, NUM_ACTIONS, ACTIONS):
    print("During the value iteration:\n")
    num_iterations = 0
    err_ls = []
    while True:
        num_iterations += 1
        nextU = constructU(NUM_ROW, NUM_COL)#[[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == NUM_COL-1) or (r == c == 1):
                    continue
                nextU[r][c] = max([calculateU(U, r, c, action, REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS) for action in range(NUM_ACTIONS)]) # Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        printEnvironment(U, NUM_ROW, NUM_COL)
        err_ls.append(error)
        if error < MAX_ERROR:
            break
    #print("# of iternations is " + str(num_iterations))
    return num_iterations, err_ls, U

# Get the optimal policy from U
def getOptimalPolicy(U, NUM_ROW, NUM_COL, NUM_ACTIONS, REWARD, DISCOUNT, ACTIONS):
    policy = [[-1] * NUM_COL for i in range(NUM_ROW)]
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            if (r <= 1 and c == NUM_COL-1) or (r == c == 1):
                continue
            # Choose the action that maximizes the utility
            maxAction, maxU = None, -float("inf")
            for action in range(NUM_ACTIONS):
                u = calculateU(U, r, c, action, REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS)
                if u > maxU:
                    maxAction, maxU = action, u
            policy[r][c] = maxAction
    return policy

# Perform some simplified value iteration steps to get an approximation of the utilities
def policyEvaluation(policy, U, NUM_ROW, NUM_COL, NUM_ACTIONS, MAX_ERROR, DISCOUNT, REWARD, ACTIONS):
    while True:
        nextU = constructU(NUM_ROW, NUM_COL)#[[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0]]
        error = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == NUM_COL-1) or (r == c == 1):
                    continue
                nextU[r][c] = calculateU(U, r, c, policy[r][c], REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS) # simplified Bellman update
                error = max(error, abs(nextU[r][c]-U[r][c]))
        U = nextU
        if error < MAX_ERROR:
            break
    return U

def policyIteration(policy, U, NUM_ROW, NUM_COL, NUM_ACTIONS, DISCOUNT, REWARD, ACTIONS, MAX_ERROR):
    print("During the policy iteration:\n")
    num_iter = 0
    while True:
        num_iter += 1
        U = policyEvaluation(policy, U, NUM_ROW, NUM_COL, NUM_ACTIONS, MAX_ERROR, DISCOUNT, REWARD, ACTIONS)
        unchanged = True
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                if (r <= 1 and c == NUM_COL-1) or (r == c == 1):
                    continue
                maxAction, maxU = None, -float("inf")
                for action in range(NUM_ACTIONS):
                    u = calculateU(U, r, c, action, REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS)
                    if u > maxU:
                        maxAction, maxU = action, u
                if maxU > calculateU(U, r, c, policy[r][c], REWARD, DISCOUNT, NUM_ROW, NUM_COL, ACTIONS):
                    policy[r][c] = maxAction # the action that maximizes the utility
                    unchanged = False
        if unchanged:
            break
        #printEnvironment(policy, NUM_ROW, NUM_COL)
    return num_iter, policy, U