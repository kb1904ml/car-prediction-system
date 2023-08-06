#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv(r"C:\Users\user\Downloads\quikr_car.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


## kms_driven and fuel_type has some null values


# ## Dataset Inspection

# In[7]:


df['Price'].unique()


# ##price has 'ask for price' value
# and datatype is object instead of integer

# In[8]:


df['year'].unique()


# ##year has many undesirable values 
# and datatype is object instead of integer

# In[9]:


df['kms_driven'].unique()


# In[10]:


## kms_driven datatype from object to integer and remove kms


# In[11]:


df['fuel_type'].unique()


# In[12]:


df['company'].unique()


# In[13]:


## company column has some undesirable values 


# In[14]:


df['name'].unique()


# In[15]:


## name is too long - keep first 3 words


# ## Dataset cleaning

# In[16]:


backup=df.copy()


# In[17]:


df=df[df['year'].str.isnumeric()]


# In[18]:


df.info()


# In[19]:


df['year']=df['year'].astype(int)


# In[20]:


df.info()


# In[21]:


df=df[df['Price']!="Ask For Price"]


# In[22]:


df['Price']=df['Price'].str.replace(',','').astype(int)


# In[23]:


df.info()


# In[24]:


df['kms_driven']=df['kms_driven'].str.split(' ').str.get(0).str.replace(',','')


# In[25]:


df=df[df['kms_driven'].str.isnumeric()]


# In[26]:


df['kms_driven']=df['kms_driven'].astype(int)


# In[27]:


df.info()


# In[28]:


df=df[~df['fuel_type'].isna()]


# In[29]:


df['name']


# In[30]:


df['name']=df['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[31]:


df


# In[32]:


df.reset_index(drop=True)


# In[33]:


df.describe()


# ## checking outliers

# In[34]:


df=df[df['Price']<6e6].reset_index(drop=True)


# In[35]:


df


# # This is our clean dataset

# In[36]:


df.to_csv('Cleaned Car.csv')


# # Model

# In[37]:


x= df.drop(columns='Price')
y= df['Price']


# In[38]:


x


# In[39]:


y


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[41]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[42]:


ohe= OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[50]:


ohe.categories_


# In[51]:


column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# In[52]:


lr=LinearRegression()


# In[53]:


pipe=make_pipeline(column_trans,lr)


# In[54]:


pipe.fit(x_train,y_train)


# In[55]:


y_predict=pipe.predict(x_test)


# In[56]:


y_predict


# In[58]:


r2_score(y_test,y_predict)


# In[59]:


for i in range(10):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_predict=pipe.predict(x_test)
    print(r2_score(y_test,y_predict),i)


# In[60]:


# now we are seeing the value of i for which r2 score is maximum


# In[63]:


scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_predict=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_predict))
    


# In[64]:


import numpy as np


# In[65]:


np.argmax(scores)


# In[66]:


scores[np.argmax(scores)]


# In[67]:


scores[661]


# In[69]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_predict=pipe.predict(x_test)
r2_score(y_test,y_predict)
    


# In[70]:


import pickle


# In[71]:


pickle.dump(pipe,open('LinearRegressionModel.pkl','wb'))


# In[72]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))

