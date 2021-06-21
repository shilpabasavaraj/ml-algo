#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[4]:


boston


# In[5]:


# feature names in the data set is columns


# In[6]:


# transform dataset into dataframe


df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)


# In[7]:


# statics of the dataset
df_x.describe()


# In[8]:


# intialise the linear regression 
reg = linear_model.LinearRegression()


# In[9]:


x_train, x_test,y_train,y_test = train_test_split(df_x,df_y,test_size = 0.33,random_state = 43)


# In[10]:


# train the model 
reg.fit(x_train,y_train)


# In[11]:


# print the co effiecients( weights) for each column

print(reg.coef_)


# In[12]:


# print the prediction on test data

y_pred = reg.predict(x_test)


# In[13]:


# print the actual values
print(y_test)


# In[14]:


print(y_pred - y_test)


# In[15]:


# check the model performance using mse

print(np.mean((y_pred - y_test)**2))


# In[16]:


# we can also use sklearn to check
from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test,y_pred))


# In[ ]:





# In[ ]:




