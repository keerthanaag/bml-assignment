#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


data=[[np.nan,8,9,50000],[np.nan,8,6,45000],[5,6,7,60000],[2,10,10,65000],[7,9,6,70000],[3,7,10,62000],[10,np.nan,7,72000],[11,7,8,80000]]


# In[5]:


df=pd.DataFrame(data,columns=['experience','test_score','interview_score','salary'])
df


# In[7]:


import math
median_experience = math.floor(df.experience.median())
print(median_experience)


# In[8]:


median_test_score = math.floor(df.test_score.median())
median_test_score


# In[9]:


df.experience = df.experience.fillna(median_experience)
df


# In[10]:


df.test_score = df.test_score.fillna(median_test_score)
df


# In[12]:


r= linear_model.LinearRegression()
r.fit(df[['experience','test_score','interview_score']],df.salary)


# In[13]:


r.coef_


# In[14]:


r.intercept_


# In[15]:


r.predict([[2,9,6]])


# In[16]:


2813.00813008*2+1333.33333333*9+2926.82926829*6+11869.91869918698


# In[18]:


r.predict([[12,9,10]])


# In[ ]:




