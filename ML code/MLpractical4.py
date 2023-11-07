#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
df = pd.read_csv(r"C:\college ML\sms+spam+collection\SMSSpamCollection", sep = '\t', names= ['label', 'message'])
df


# In[52]:


df.info()


# In[53]:


df.isnull().sum()


# In[54]:


df.describe()


# In[55]:


df.groupby('label').describe()


# In[57]:


df['spam'] = df['label'].apply( lambda x:1 if x== 'spam' else 0)
df


# In[61]:


#input
x = df['message']
#output
y = df['spam']
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_count = v.fit_transform(x)


# In[62]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_count, y)


# In[65]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)


# In[66]:


y_pred = model.predict(x_train)


# In[68]:


model.score(x_test, y_test)


# In[ ]:




