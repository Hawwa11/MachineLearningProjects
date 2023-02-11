#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary modules
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv(r'C:\INTI\Semester 7\PRG4206 Machine Learning\Group Assignment\depress.csv')


# In[3]:


df.head()


# In[4]:


#preprocessing data
#take out rows with null values
df = df.dropna()

#separate into data (fetures only) and target (goal)
data = df.drop(columns=['depressed'])  #features only. remove the target column
target = df['depressed'].values   #the target only aka 'depressed or not?'


# In[5]:


from sklearn.model_selection import train_test_split

#dividing data into train and test
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

print("Shape of X_test: {}".format(X_test.shape))
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_test: {}".format(y_test.shape))
print("Shape of y_train: {}".format(y_train.shape))


# In[6]:


#building random forest model with hyperparameters

#generating random estimators
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,4,6,8]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[7]:


# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)


# In[8]:


#create model  
rf_Model = RandomForestClassifier()


# In[9]:


#using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid,n_iter =100, cv = 10, verbose=2, n_jobs = 1)


# In[10]:


rf_RandomGrid.fit(X_train, y_train)


# In[11]:


rf_RandomGrid.best_params_


# In[12]:


print('With RandomizedSearch')
print (f'Train Accuracy - : {rf_RandomGrid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_RandomGrid.score(X_test,y_test):.3f}')


# In[13]:


#use the forest's predict method on the test data
prediction = rf_RandomGrid.predict(X_test)

print(prediction)


# In[14]:


#import scikit-learn metrics module for accuract testing
from sklearn import metrics

#model accuracy, how often is the classifier correct?
#compare predicted values to the actual target value in y_test set

print("Accuracy (Optimized - RandomizedSearch): ", metrics.accuracy_score(y_test, prediction))


# In[15]:


#Evaluation metrics
#Constructing the confusion matrix.
from sklearn. metrics import confusion_matrix
confusion_matrix(y_test, prediction)


# In[16]:


from sklearn.tree import export_graphviz
import pydot
import pydotplus

feature_list = list(data.columns)

tree = rf_RandomGrid.best_estimator_[5]
export_graphviz(tree, out_file = "tree.dot", filled = True, feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('RandomForestOptimizedRandomizedSearch.png')

#display the image
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
img=pltimg.imread('RandomForestOptimizedRandomizedSearch.png')
imgplot = plt.imshow(img)
plt.show()


# In[17]:


import mglearn

mglearn.discrete_scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train)


# In[18]:


mglearn.discrete_scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test)


# In[19]:


mglearn.discrete_scatter(prediction, prediction, y_test)


# In[20]:


import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(rf_RandomGrid.best_estimator_[5], 
                           feature_names=feature_list,
                           class_names="target", 
                           filled=True, impurity=True, 
                           rounded=True)

graph = graphviz.Source(dot_data, format='png')
graph

