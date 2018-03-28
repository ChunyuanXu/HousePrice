# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

train_data = pd.read_csv('../Data/train.csv')
test_data = pd.read_csv('../Data/test.csv')

data_column = train_data.columns

# 单独处理house price
price_column = data_column[-1]
price_data = train_data[price_column].values
train_data = train_data.drop(price_column, axis=1)
id_column = data_column[0]
train_data = train_data.drop(id_column, axis=1)

test_data = test_data.drop(id_column, axis=1)

# 删除缺省数值率高于50%的列
null_idx = 1+np.where(np.sum(train_data.isnull()) > (0.5*train_data.shape[0]))[0]
null_column = data_column[null_idx]
print 'higher than 50% lost data:', null_column
train_data = train_data.drop(null_column, axis=1)
data_column = train_data.columns  # update data_column

test_data = test_data.drop(null_column, axis=1)

train_data_copy = train_data.copy()

# 与year相关的数据
year_related_columns = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
train_data_copy['BuiltTime'] = train_data['YrSold'] - train_data['YearBuilt']
train_data_copy['RemodAddTime'] = train_data['YrSold'] - train_data['YearRemodAdd']
train_data_copy['GarageBltTime'] = train_data['YrSold'] - train_data['GarageYrBlt']
train_data_copy = train_data_copy.drop(year_related_columns, axis=1)
data_column = train_data_copy.columns  # update data_column

test_data['BuiltTime'] = test_data['YrSold'] - test_data['YearBuilt']
test_data['RemodAddTime'] = test_data['YrSold'] - test_data['YearRemodAdd']
test_data['GarageBltTime'] = test_data['YrSold'] - test_data['GarageYrBlt']
test_data = test_data.drop(year_related_columns, axis=1)

# 将特征数值划分为数值、非数值，分别处理
data_types = train_data_copy.dtypes
num_type_idx = data_types != object
obj_type_idx = data_types == object

# 数值特征 使用均值填补缺省值
num_column = data_column[num_type_idx]
print('length of num_column', len(num_column))
num_data = train_data_copy[num_column].values
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values=np.nan, strategy='mean')
num_data = imr.fit_transform(num_data)

num_data_test = test_data[num_column].values
num_data_test = imr.transform(num_data_test)

## visualize features in the dataset
#train_data_display = pd.DataFrame(data=num_data, columns=num_column)
#import matplotlib.pyplot as plt
#import seaborn as sns
##sns.set(style='whitegrid', context='notebook')
#display_column = data_column[num_type_idx][-8:-3]
##sns.pairplot(train_data_display[display_column], size=2.5)
##plt.show()
#
#cm = np.corrcoef(train_data_display[display_column].values.T)
#sns.set(font_scale=1.5)
#hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', \
#                annot_kws={'size':15}, yticklabels=display_column, xticklabels=display_column)
#plt.show()

# 非数值特征
train_data_class = train_data_copy.copy()
train_data_class.drop(num_column, axis=1, inplace=True)

test_data.drop(num_column, axis=1, inplace=True)

#for col in train_data_class.columns:
#    print train_data_class[col].value_counts()
# 特征极其不均衡的数据，将其删除
unimbalanced_feature = ['Street', 'Utilities', 'Heating', 'Condition2', 'RoofMatl']
train_data_class.drop(unimbalanced_feature, axis=1, inplace=True)

test_data.drop(unimbalanced_feature, axis=1, inplace=True)

train_test_data = pd.concat([train_data_class, test_data])
# 将标签数据改为0、1、2、。。。
for col in train_test_data.columns:
    col_value = pd.factorize(train_test_data[col])[0]
    train_data_class[col] = col_value[:train_data_class.shape[0]]
    test_data[col] = col_value[train_data_class.shape[0]:]
# 最频繁项填补缺省值
class_data = train_data_class.values
imr_class = Imputer(missing_values=-1, strategy='most_frequent')
class_data = imr_class.fit_transform(class_data)

class_data_test = test_data.values
class_data_test = imr_class.transform(class_data_test)

# 使用OneHot编码处理
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
class_data_oh = ohe.fit_transform(class_data).toarray()

class_data_test_oh = ohe.transform(class_data_test).toarray()

# 数值型数据归一化
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
num_data_norm = mms.fit_transform(num_data)

num_data_test_norm = mms.transform(num_data_test)

# 保存数据
np.save('../MidData/price_label.npy', price_data)
np.save('../MidData/num_data.npy', num_data)
np.save('../MidData/class_data_oh.npy', class_data_oh)
np.save('../MidData/class_data.npy', class_data)
np.save('../MidData/num_data_norm.npy', num_data_norm)

np.save('../MidData/num_data_test.npy', num_data_test)
np.save('../MidData/num_data_test_norm.npy', num_data_test_norm)
np.save('../MidData/class_data_test.npy', class_data_test)
np.save('../MidData/class_data_test_oh.npy', class_data_test_oh)








