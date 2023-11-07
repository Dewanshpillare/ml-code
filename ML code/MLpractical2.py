#!/usr/bin/env python
# coding: utf-8

# In[2]:


#PROBLEM 1
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\college ML\archive (1)\temperatures.csv")

df.isnull().sum()# as there is no null value we can proceed


# In[3]:


df.info()




# In[15]:


x = df['YEAR']
y = df['ANNUAL']
import matplotlib.pyplot as plt
plt.title('Temprature Graph')
plt.xlabel('YEAR')
plt.ylabel('ANNUAL')
plt.scatter(x,y)


# In[45]:


x.shape #1d


# In[50]:


#x = x.to_numpy() First do this. Then only it will work
x =x.reshape(-1,1)
x.shape




# In[51]:


from sklearn.model_selection import train_test_split #splitting data
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.25)
from sklearn.linear_model import LinearRegression # import model
model = LinearRegression()#create object
model.fit(x_train,y_train)#train model


# In[59]:


model.coef_


# In[61]:


model.intercept_


# In[62]:


predicted = model.predict(x_test) #testing model using test data


# In[75]:


#PROBLEM 2
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,predicted)


# In[76]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predicted)


# In[78]:


from sklearn.metrics import r2_score
r2_score(y_test, predicted)


# In[79]:


#PROBLEM 3
import matplotlib.pyplot as plt
plt.title('Temprature Graph')
plt.xlabel('YEAR')
plt.ylabel('ANNUAL')
plt.scatter(x,y)
plt.plot(x_test, predicted,color = "g")


# In[84]:


model.score(x,y)


# In[ ]:




