# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 16:54:35 2018

@author: yuan
"""

import numpy as np
import joblib
import pandas as pd

testData_num = np.load('../MidData/num_data_test_norm.npy')
testData_class = np.load('../MidData/class_data_test.npy')

testData = np.concatenate((testData_num, testData_class), axis=1)
rfr = joblib.load('../Model/model_rfr_norm.pkl')
pred = rfr.predict(testData)*10000
id_list = np.array(range(testData.shape[0]))+1461
predDF = pd.DataFrame(id_list, columns=['Id'])
predDF['SalePrice'] = pred

predDF.to_csv('../Submission/my_rfr_withNormData.csv', index=False)

