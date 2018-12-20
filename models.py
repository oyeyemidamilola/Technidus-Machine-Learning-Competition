# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 07:58:50 2018

@author: USER
"""

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

# supervised learning algorithms

def adaboosted_decision_tree_regressor(X,y):
    
    """
    This function implements an adaboosted decision tree regressor
    X : numerical features -- predictors
    y : the target variables
    """
    # hyper-paramters
    max_depth = 10
    n_estimators = 100
    rand_state = 7
    
    model = AdaBoostRegressor(DecisionTreeRegressor(max_depth=max_depth),n_estimators=n_estimators,random_state = rand_state)
    
    return model