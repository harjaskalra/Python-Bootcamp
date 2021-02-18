#!/usr/bin/env python
# coding: utf-8

# # Python 3
# For this tutorial we'll be using the Iris dataset from sklearn. 
# 
# In this notebook we will:
# 1. Import required modules and dataset
# 2. Define multiple Classification models
# 3. Fit the data to our models
# 4. Use our trained models to predict a class label 
# 5. Evaluate our models and chose the best performing model 
# 
# 

# In[9]:


#Import Pandas to your workspace
import pandas as pd


# In[16]:


#Read the "features.csv" file and store it into a variable

features = pd.read_csv('data/features.csv')


# In[17]:


#Display the first few rows of the DataFrame

features.head()


# <h1>groupby()</h1>
# 
# <ul>
#     <li>groupby combines 3 steps all in one function:
#         <ol>
#             <li>Split a DataFrame</li>
#             <li>Apply a function</li>
#             <li>Combine the results</li>
#         </ol>
#     </li>
#     <li>groupby must be given the name of the column to group by as a string</li>
#     <li>The column to apply the function onto must also be specified, as well as the function to apply</li>
# </ul>

# <img src="images/groupbyviz.jfif"/>

# In[23]:


#Apply groupby to the Year and Month columns, calculating the mean of the CIP

year_CPI = features.groupby(['Year', 'Month'])['CPI'].mean().reset_index()

year_CPI.head(5)


# In[25]:


#Groupby returns a DataFrame, so we have access to all the same methods we saw earlier
year_CPI.sort_values(by='Year', ascending = False)


# In[26]:


#Read the "stores.csv" file and store it into a variable called stores

stores = pd.read_csv("data/stores.csv")


# In[27]:


#Display the first few rows of the stores DataFrame

stores.head()


# In[31]:


stores.dtypes


# In[37]:


#Convert the values in the 'Type' column from upper to lower case 

stores['Type'] = stores['Type'].str.lower()
stores.head()


# In[44]:


stores.columns


# In[47]:


#Rename the 'Size' column to 'Area'

stores.rename(columns={'Size': 'Area'}, inplace=True)


# In[48]:


#Display the first few rows to verify changes
stores.head()


# <h1>merge()</h1>
# 
# <ul>
#     <li>Merge two DataFrames along common columns</li>
#     <li>Must be provided the DataFrame to merge with, as well as the names of the common columns</li>
#     <li>Will merge and map rows where the values in both DataFrames are equal</li>
# </ul>

# <img src="images/mergetypes.png"/>

# <img src="images/mergeinner.png"/>

# In[50]:


features.head()


# In[51]:


stores.head()


# In[52]:


#Merge the stores DataFrame into the features DataFrame on the Stores column
df_merged = features.merge(stores, on='Store')


# In[53]:


df_merged.head()


# In[ ]:





# In[ ]:


#Display a few rows to verify changes


# In[54]:


#Export the final version of our DataFrame to a .csv file named "final_data.csv" 
df_merged.to_csv('data/final_data.csv', header=True)


# <h1>Part 2 - Machine Learning</h1>

# In[55]:


#Import libraries we will need

# numpy
import numpy

# scikit-learn
import sklearn

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import datasets

from IPython.display import display

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# In[ ]:





# In[63]:


#2.2 Load Dataset

dataset = datasets.load_iris()
feature_names = dataset['feature_names']
iris_data = pd.DataFrame(data=dataset['data'], columns = feature_names)
target = pd.DataFrame(data=dataset['target'], columns=['class'])


# In[57]:


display(dataset)


# In[61]:


dataset.feature_names


# In[64]:


#Dimensions of Dataset

print(iris_data.shape)


# In[65]:


#Peek at the Data

iris_data.head()


# In[66]:


#Statistical Summary

iris_data.describe()


# In[67]:


#Class Distribution - value_counts function to see number of each class

target['class'].value_counts()


# In[68]:


#Data Visualization
#Using the plot() function, we can make boxplots by simply specifying the kind of plot
iris_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)

plt.show()


# In[70]:


#Histograms

iris_data.hist()
plt.show


# In[71]:


# Multivariate Plots
# scatter plot matrix

scatter_matrix(iris_data)
plt.show


# In[78]:


#Create the Train and Test set
X = iris_data.values
Y = target.values
test_size = 0.20
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, 
                                                                    test_size = test_size, 
                                                                    random_state=seed)


#We use train_test_split to shuffle and divide our data into our train and test sets


# In[ ]:


print


# <img src='images/train_test_split.png'/>

# <img src='images/mlprocess.png'/>

# In[ ]:


#Verify our split


# In[ ]:





# In[80]:


#Create an instance of our algorithm (model)
LDA = LinearDiscriminantAnalysis()


# In[81]:


#Feed our training data to our model

LDA.fit(X_train, Y_train)


# In[82]:


#Test our model on the test set

LDA.score(X_test, Y_test)


# In[83]:


X_test[:5]


# In[84]:


Y_test[:5]


# In[85]:


#Use predict() to obtain prediction from our model on data points

LDA.predict([[5.9, 3. , 5.1, 1.8]])


# In[93]:


for point in X_test:
    
    prediction = LDA.predict([point])
    
    print(f"Class value of {prediction}")


# In[ ]:




