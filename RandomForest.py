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


#split the dataset into 80% training 20% testing
from sklearn.model_selection import train_test_split

#test_size=0.2 means 20% of data goes into testing
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0)

print("Shape of X_test: {}".format(X_test.shape))
print("Shape of X_train: {}".format(X_train.shape))
print("Shape of y_test: {}".format(y_test.shape))
print("Shape of y_train: {}".format(y_train.shape))


# In[6]:


#build random forest classifier
from sklearn import ensemble
rf_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=0)

#fit the training data into the model
rf_clf.fit(X_train, y_train)

print ("Accuracy on training set: ", (rf_clf. score(X_train, y_train)))
print ("Accuracy on testing set: ", (rf_clf. score (X_test, y_test)))


# In[7]:


#use the forest's predict method on the test data
prediction = rf_clf.predict(X_test)

print(prediction)


# In[8]:


#import scikit-learn metrics module for accuracy testing
from sklearn import metrics

#model accuracy, how often is the classifier correct?
#compare predicted values to the actual target value in y test set

print("Accuracy: ", metrics.accuracy_score(y_test, prediction))


# In[9]:


from sklearn.tree import export_graphviz
import pydot
import pydotplus

feature_list = list(data.columns)

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


# In[10]:


#Evaluation metrics
#Constructing the confusion matrix.
from sklearn. metrics import confusion_matrix
confusion_matrix(y_test, prediction)


# In[11]:


import mglearn

mglearn.discrete_scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train)


# In[12]:


mglearn.discrete_scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], y_test)


# In[13]:


mglearn.discrete_scatter(prediction, prediction, y_test)


# In[14]:


import graphviz
from sklearn.tree import export_graphviz


dot_data = export_graphviz(rf_clf.estimators_[5], 
                           feature_names=feature_list,
                           class_names="target", 
                           filled=True, impurity=True, 
                           rounded=True)

graph = graphviz.Source(dot_data, format='png')
graph


# In[15]:


from sklearn import tree

# This may not the best way to view each estimator as it is small
fn=feature_list
cn="depressed"
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf_clf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')

