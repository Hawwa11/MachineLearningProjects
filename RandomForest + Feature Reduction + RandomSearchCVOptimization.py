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

print("\nSize of our data: {}".format(df.shape))


# In[5]:


# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1409, n_features=22, n_informative=5, n_redundant=5, random_state=1)
# define the model
model = RandomForestClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

def select_features(X_train, y_train, X_test):
    # configure to select a subset of features
    fs = SelectFromModel(RandomForestClassifier(n_estimators=1000), max_features=10)
    # learn relationship from training data
    fs.fit(X_train, y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)
    # transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


# In[7]:


# define the dataset
X, y = make_classification(n_samples=1409, n_features=22, n_informative=5, n_redundant=5, random_state=1)
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


# In[8]:


#for visualization
#data = pd.merge(X_train_fs, y_train)

dftrain = pd.DataFrame(X_train_fs)
dftest = pd.DataFrame(X_test_fs)


# In[9]:


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


# In[10]:


# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)


# In[11]:


#create model  
rf_Model = RandomForestClassifier()


# In[12]:


#using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = rf_Model, param_distributions = param_grid,n_iter =100, cv = 10, verbose=2, n_jobs = 1)


# In[13]:


rf_RandomGrid.fit(X_train_fs, y_train)


# In[14]:


rf_RandomGrid.best_params_


# In[15]:


print('With RandomizedSearch')
print (f'Test Accuracy - : {rf_RandomGrid.score(X_test_fs,y_test):.3f}')


# In[16]:


#use the forest's predict method on the test data
prediction = rf_RandomGrid.predict(X_test_fs)

print(prediction)


# In[17]:


#import scikit-learn metrics module for accuract testing
from sklearn import metrics

#model accuracy, how often is the classifier correct?
#compare predicted values to the actual target value in y_test set

print("Accuracy (Optimized - RandomizedSearch): ", metrics.accuracy_score(y_test, prediction))


# In[18]:


#Evaluation metrics
#Constructing the confusion matrix.
from sklearn. metrics import confusion_matrix
confusion_matrix(y_test, prediction)


# In[19]:


from sklearn.tree import export_graphviz
import pydot
import pydotplus

feature_list = list(dftrain)

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


# In[20]:


import mglearn

mglearn.discrete_scatter(dftrain.iloc[:, 0], dftrain.iloc[:, 1], y_train)


# In[21]:


mglearn.discrete_scatter(dftest.iloc[:, 0], dftest.iloc[:, 1], y_test)


# In[22]:


mglearn.discrete_scatter(prediction, prediction, y_test)


# In[23]:


import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(rf_RandomGrid.best_estimator_[5], 
                           feature_names=feature_list,
                           class_names="target", 
                           filled=True, impurity=True, 
                           rounded=True)

graph = graphviz.Source(dot_data, format='png')
graph

