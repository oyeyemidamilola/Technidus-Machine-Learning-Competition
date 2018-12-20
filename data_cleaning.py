# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 05:09:39 2018

@author: USER
"""

import pandas as pd
import numpy as np
import os


# read in raw data
def data_transform(path):
    
    if not os.path.exists(path):
        raise FileNotFoundError('Dataset does not exist')
    columns_to_drop = ['Title','FirstName','MiddleName','LastName','Suffix',
                       'AddressLine1','AddressLine2','PostalCode','PhoneNumber','BirthDate']
    dataset = pd.read_csv(path)
    dataset = dataset.drop(columns = columns_to_drop);
    dataset.columns = ['C_Id','city','SP','CRN','Edu','Ocu','Gender','MS','HO','NCO','NChil','TChil','YI','AMS','BB']
    
    # normalize the income value
    max_year_income = max(dataset['YI'])
    min_year_income = min(dataset['YI'])
    dataset['YI'] = (max_year_income - dataset['YI'])/(max_year_income-min_year_income)
    
    return dataset


