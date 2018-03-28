# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 21:54:15 2018

@author: yuan
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

data_num = np.load('../MidData/num_data_norm.npy')
data_class = np.load('../MidData/class_data.npy')
price_label = np.load('../MidData/price_label.npy')
price_label = price_label/10000.

# concat data_num and data_class(_oh)
data = np.concatenate((data_num, data_class), axis=1)

# train & validation data split
X_train, X_val, y_train, y_val = train_test_split(data, price_label, test_size=0.1, random_state=1)

rfr = RandomForestRegressor(n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
rfr.fit(X_train, y_train)

joblib.dump(rfr, '../Model/model_rfr_norm.pkl')
y_train_pred = rfr.predict(X_train)
y_val_pred = rfr.predict(X_val)
print('MSE train: %.3f, test: %.3f' %(mean_squared_error(y_train, y_train_pred), \
                                      mean_squared_error(y_val, y_val_pred)))
print('r2 score train: %.3f, test: %.3f' %(r2_score(y_train, y_train_pred), \
                                           r2_score(y_val, y_val_pred)))
#MSE train: 1.676, test: 6.923
#r2 score train: 0.972, test: 0.903

plt.scatter(y_train_pred, y_train_pred-y_train, c='black', marker='o', s=35, alpha=0.5, label='Train data')
plt.scatter(y_val_pred, y_val, c='lightgreen', marker='s', s=35, alpha=0.7, label='Val data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=100, lw=2, color='red')
plt.xlim([0, 100])
plt.show()