#!/usr/bin/env python
# coding: utf-8

# In[1]:


#: k-means:dbs scan / k means clustering :


# In[2]:


#C:\Users\Shilpa\Downloads


# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv("C://Users//Shilpa//Downloads//train.csv")


# In[3]:


train.head()


# In[4]:


# missing values:
train.isnull()


# In[5]:


sns.heatmap(train.isnull(),xticklabels = True,cmap = 'viridis')# yticklabels = False;


# In[6]:


sns.set_style('whitegrid')
sns.countplot( x = 'Survived', data = train)


# In[7]:


sns.set_style('whitegrid')
   sns.countplot(x = 'Survived', hue = 'Sex', data = train, palette= 'RdBu_r')


# In[8]:


sns.set_style('whitegrid')
sns.countplot(x = 'Survived', hue = 'Pclass', data = train, palette = 'rainbow')


# In[9]:


#sns.countplot(x = '', y = '', dat = 'df')


# In[10]:


sns.distplot(train['Age'].dropna(), kde =False, color = 'darkred', bins = 40)


# In[11]:


sns.countplot(x = 'SibSp', data = train)


# In[12]:


train['Fare'].hist( color = 'green', bins = 40,figsize = (8,4))


# In[13]:



plt.figure(figsize = (12,7))
sns.boxplot(x = 'Pclass', y = 'Age', data = train, palette = 'winter')


# In[14]:


# we can see the class1(rich paasengers)tend to be older, which makes sense,imputing the mean value for age column


# In[15]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
            
        else:
             return 24
            
    else:
        return Age


# In[16]:


train['Age'] = train[['Age','Pclass']].apply(impute_age, axis = 1)


# In[17]:



sns.heatmap(train.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')


# In[18]:


# age does not have any nullvaluse now, replaced  by the mean


# In[19]:



train.drop('Cabin', axis = 1,inplace = True)


# In[20]:


train.head()

pd.get_dummies(train['Embarked'],drop_first = True).head()
# In[21]:


pd.get_dummies(train['Embarked'],drop_first = True).head()


# In[22]:


sex  = pd.get_dummies(train['Sex'],drop_first = True)

                    
                    


# In[23]:


embark = pd.get_dummies(train['Embarked'],drop_first = True)


# In[24]:


train.drop(['Sex','Embarked','Name','Ticket'], axis = 1,inplace = True)


# In[25]:


train.head()


# In[26]:


train = pd.concat([train,sex,embark],axis= 1)


# In[27]:


train.head()


# In[28]:


# building the logistic regression:


# In[29]:


train.drop('Survived', axis = 1).head()
 # train.drop('Survived', axis = 1).head()


# In[30]:


train['Survived'].head()


# In[48]:


from sklearn.model_selection import train_test_split


# In[49]:


X_train,X_test,y_train,y_test = train_test_split(train.drop('Survived',axis = 1),train['Survived'],test_size = 0.30,
                                                random_state = 101)


# In[50]:


train_test_split(train.drop('Survived',axis = 1),train['Survived'],test_size = 0.30, random_state = 101)


# In[59]:


from sklearn.linear_model import LogisticRegression


# In[60]:


logmodel = LogisticRegression(max_iter = 500);
logmodel.fit(X_train,y_train)


# In[61]:


predictions = logmodel.predict(X_test)


# In[62]:


from sklearn.metrics import confusion_matrix


# In[63]:


accuracy = confusion_matrix(y_test,predictions)


# In[64]:


accuracy


# In[65]:


from sklearn.metrics import accuracy_score


# In[67]:


accuracy = accuracy_score(y_test,predictions)
accuracy


# In[ ]:




