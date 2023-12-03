#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[6]:


data = pd.read_csv("E:/Python docs/Position_Salaries.csv")
data


# In[8]:


X = data['Level'].values.reshape(-1,1)
y = data['Salary'].values.reshape(-1,1)


# In[9]:


sns.scatterplot(x = 'Level', y = 'Salary', data = data)


# In[10]:


##### Simple Linear Regression ---
from sklearn.linear_model import LinearRegression


# In[11]:


lin = LinearRegression()
print("The model is loaded")


# In[12]:


lin.fit(X,y)
print("Training Completed")


# In[13]:


plt.figure(figsize=(5,3))
plt.scatter(X,y,color = 'blue')
plt.plot(X,lin.predict(X), color = 'red')


# In[14]:


#### Polynomial Regression - Features will get transform into polynomial Fetaures
#### (degree = 3)


# In[15]:


from sklearn.preprocessing import PolynomialFeatures


# In[16]:


poly = PolynomialFeatures(degree = 3)
print("Loaded the Polynomial Features")


# In[17]:


X_poly = poly.fit_transform(X)
X_poly


# In[18]:


poly.fit(X_poly,y)
print("Polynomial Features is fitted and Trained")


# In[19]:


lin2 = LinearRegression()


# In[20]:


lin2.fit(X_poly,y)
print("The Model is trained using Polynomial Features")


# In[21]:


plt.figure(figsize=(5,3))
plt.scatter(X,y,color = 'blue')
plt.plot(X,lin2.predict(X_poly), color = 'red')


# In[22]:


from sklearn.metrics import r2_score


# In[23]:


r2_score(y,lin.predict(X)) #### rsquare value for Linear Regression


# In[24]:


r2_score(y,lin2.predict(X_poly)) #### rsquare value for Polynomial Regression


# In[ ]:





# In[ ]:




