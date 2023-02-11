#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the necessary modules
import pandas as pd
import numpy as np

#read the dataset into dataframe
df = pd.read_csv(r'C:\Users\Souffle\Downloads\depress.csv')

print("\nDataframe: \n{}".format(df))
print("\nSize of our data: {}".format(df.shape))


# In[2]:


df.head()


# In[3]:


#preprocessing data
#take out rows with null values
df = df.dropna()

print("\nSize of our data: {}".format(df.shape))


# In[4]:


#separate into Data (features only) and Target (our goal)
data = df.drop(columns=['depressed'])  #features only. remove the target column
target = df['depressed'].values   #the target only aka 'depressed or not?'

print(data)
print(target)


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


# evaluation of a model using 5 features chosen with random forest importance
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
# feature selection
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

print("Shape of X_test: {}".format(X_test.shape))
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_test: {}".format(y_test.shape))
print("Shape of y_train: {}".format(y_train.shape))

# feature selection
X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)


# In[8]:


#for visualization
#data = pd.merge(X_train_fs, y_train)

dftrain = pd.DataFrame(X_train_fs)
dftest = pd.DataFrame(X_test_fs)


# In[9]:


#build random forest classifier
from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=0)


# In[10]:


#fit the training data into the model
rf_clf.fit(X_train_fs, y_train)


# In[11]:


print ("Accuracy on testing set: ", (rf_clf. score (X_test_fs, y_test)))


# In[12]:


#use the forest's predict method on the test data
prediction = rf_clf.predict(X_test_fs)

print(prediction)


# In[13]:


#import scikit-learn metrics module for accuracy testing
from sklearn import metrics

#model accuracy, how often is the classifier correct?
#compare predicted values to the actual target value in y test set

print("Accuracy: ", metrics.accuracy_score(y_test, prediction))


# In[14]:


#Evaluation metrics
#Constructing the confusion matrix.
from sklearn. metrics import confusion_matrix
confusion_matrix(y_test, prediction)


# In[15]:


from sklearn.tree import export_graphviz
import pydot
import pydotplus

feature_list = list(dftrain)

tree = rf_clf.estimators_[5]

export_graphviz(tree, out_file = "tree.dot", filled = True, feature_names = feature_list, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('tree.dot')

graph.write_png('RandomForest.png')

#display the image
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
img=pltimg.imread('RandomForest.png')
imgplot = plt.imshow(img)
plt.show()


# In[16]:


import mglearn

mglearn.discrete_scatter(dftrain.iloc[:, 0], dftrain.iloc[:, 1], y_train)


# In[17]:


mglearn.discrete_scatter(dftest.iloc[:, 0], dftest.iloc[:, 1], y_test)


# In[18]:


mglearn.discrete_scatter(prediction, prediction, y_test)


# In[19]:


import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(rf_clf.estimators_[5], 
                           feature_names=feature_list,
                           class_names="target", 
                           filled=True, impurity=True, 
                           rounded=True)

graph = graphviz.Source(dot_data, format='png')
graph

