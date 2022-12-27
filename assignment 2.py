#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[4]:


data=[[7,38],[23,58],[78,9],[10,9],[3,79],[49,12]]
dataset=pd.DataFrame(data,columns=['AGE','GLUCOSE LEVEl'])
dataset.set_index(np.array(range(1,7)))


# In[7]:


a=np.array(dataset['AGE'])
b=np.array(dataset['GLUCOSE LEVEl'])


# In[8]:


get_ipython().run_line_magic('matplotlib','inline')
plt.scatter(a,b)
plt.xlabel('AGE')
plt.ylabel('GLUCOSE LEVEl')


# In[10]:


a=np.array(a.reshape(-1,1))
reg=linear_model.LinearRegression()
reg.fit(a,b)


# In[11]:


reg.predict(np.array([55]).reshape(-1,1))


# In[12]:


reg.coef_


# In[13]:


reg.intercept_


# In[ ]:




