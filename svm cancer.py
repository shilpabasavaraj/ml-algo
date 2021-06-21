#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import pandas as pd
#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()

df = pd.read_csv(r"C:\Users\Shilpa\OneDrive\Desktop\ds\cancer.csv")
df.head();


# In[3]:


df.head()


# In[4]:


df['Class'].value_counts()


# In[5]:


b_df = df[df["Class"]==2][0:200]


# In[6]:


b_df.head()


# In[7]:


m_df = df[df['Class']==4][0:200]


# In[8]:


m_df.head()


# In[9]:


# help 
axes = b_df.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize', color = 'blue', label = 'benign')
m_df.plot(kind = 'scatter', x = 'Clump', y = 'UnifSize', color = 'red', label = 'malig', ax= axes)


# In[10]:


# if you want both the benign and  malignanat in the same plot then ax = axes


# In[11]:


# identifying the  unewanted rows
df.dtypes


# In[12]:


# barenuc is a object type( nota number ), we cannot appply any mathematical equations to them , so we need to
# convert them

df = df[pd.to_numeric(df['BareNuc'], errors = 'coerce').notnull()]


# In[13]:


df.dtypes


# In[14]:


# still it has object values , so we have to convert them into integers
df['BareNuc']=df['BareNuc'].astype('int')


# In[15]:


df.dtypes


# In[16]:


# skipped id and class
feature_set = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']] 


# In[17]:


df.shape


# In[18]:


# we have 9 columns as we left "class" and "id",so that is our X


# In[20]:


X = np.asarray(feature_set)
y = np.asarray(df['Class'])


# In[21]:


X[0:5]


# In[25]:


# divide the train adn test data
from sklearn.model_selection  import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size = 0.2, random_state = 42)


# In[26]:


X_train.shape


# In[27]:


y_train.shape


# In[33]:


# modelling :
from sklearn import svm
cl = svm.SVC(kernel = 'linear', gamma = 'auto', C = 2)
cl.fit(X_train,y_train)
y_predict = cl.predict(X_test)
 
# kernel can accept 4 types of functions : linear, rbf, poly and sigmoid


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))


# In[ ]:


# precision : ratio of true predictios

