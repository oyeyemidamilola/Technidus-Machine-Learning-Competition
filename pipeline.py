# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 07:44:21 2018

@author: 
"""
import pandas as pd
from data_cleaning import data_transform
from models import adaboosted_decision_tree_regressor



if __name__ == '__main__':
    
    # path to the files
     
    
    train_path = 'train_technidus.csv'
    test_path = 'test_technidus.csv'
    
    # read train and train sets from CSV files
    # features selection
    train_set = data_transform(train_path).drop(columns = ['city','SP'])
    test_set = data_transform(test_path).drop(columns = ['city','SP','AMS'])
    
    # get all categorical variables
    categorical_variables = list(train_set.columns^train_set.describe().columns)
    
    # features
    # one hot encoding of categorical variables
    X_train = pd.get_dummies(train_set,columns = categorical_variables).drop(columns=['AMS'])
    y = train_set['AMS'].values
    
    # model
    model = adaboosted_decision_tree_regressor(X_train,y)
    model.fit(X_train,y)
    
    
    # testing 
    X_test = pd.get_dummies(test_set,columns = categorical_variables)    
    y_pred = model.predict(X_test)
    
    predictions = pd.DataFrame({'CustomerID':test_set['C_Id'],'AveMonthSpend':y_pred},dtype=int)

    # export to csv
    predictions.to_csv('submission_adaboosted_decision_tree.csv')
    
    print('DONE')
    
    
    
    
    

    
    
    
    
