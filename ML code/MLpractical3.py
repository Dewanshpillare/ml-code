#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
import seaborn as sns
df = pd.read_csv(r"C:\college ML\archive (2)\Admission_Predict_Ver1.1.csv")
df


# In[51]:


df.info()


# In[52]:


df.isnull().sum()


# In[53]:


#QUE 1 Data Transformation
from sklearn.preprocessing import Binarizer
bi = Binarizer(threshold = 0.75)
df['Chance of Admit '] = bi.fit_transform(df[['Chance of Admit ']])
df


# In[54]:


#data preparation

#here the Chance of admit is output variable
#we drop the chance of admit column and store rest of table in input variable x
#chance of admit is stored in output variable y

#input
x = df.drop('Chance of Admit ', axis = 1)
#output
y = df['Chance of Admit ']


# In[55]:


#as we need only 0 and 1 in the last column we change its datatype form flot to int
y = y.astype('int')
y


# In[56]:


#printing graph of our dataset.here x is not input data. it is the parameter of method .countplot(x = none)
sns.countplot(x = y)


# In[57]:


y.shape


# In[58]:


#QUE 2 data preparation

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size= 0.25)


# In[59]:


#QUE 3 training model

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state = 0)
classifier.fit(x_train,y_train)


# In[60]:


y_pred = classifier.predict(x_test)


# In[61]:


#QUE 4 evaluating model

classifier.score(x_test, y_test)


# In[62]:


from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test,y_pred)
print('accuracy :', accuracy)


# In[64]:


print(classification_report(y_test,y_pred))


# In[66]:


ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

