#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Read the dataset
import pandas as pd

df = pd.read_csv ('/Users/uSER/Downloads/spotify_top50_2021.csv')


#Dataset Details
print("\nDataframe Shape/Size: {}".format(df.shape))
print("\nFile Name: spotify_top50_2021.csv")
print("\nFile Format: .csv")

#Print 5 records of the Dataset
df.head()


# In[2]:


# Understanding the dataset

df.info()

# Observations
# There are no missing values in the dataset, this is evident from the non-null count
# There are total 18 columns
# The range of index states there are 50 entires (0 to 49 record indexes)
# There are 3 object, 6 int64 and 9 float64 data type columns.


# In[3]:


#Prepare data

#Drop useless or extra columns
df = df.drop(["id"], axis=1)
df = df.drop(["artist_name"], axis=1)
df = df.drop(["track_id"], axis=1)
df = df.drop(["key"], axis=1)
df = df.drop(["loudness"], axis=1)
df = df.drop(["mode"], axis=1)
df = df.drop(["liveness"], axis=1)
df = df.drop(["tempo"], axis=1)
df = df.drop(["duration_ms"], axis=1)
df = df.drop(["time_signature"], axis=1)


df.head()


# In[4]:


#Prepare data cont.
from sklearn.preprocessing import MinMaxScaler

#Standarize data
scaler = MinMaxScaler()

scaler.fit(df[['popularity']])
df['popularity'] = scaler.transform(df[['popularity']])

scaler.fit(df[['danceability']])
df['danceability'] = scaler.transform(df[['danceability']])

scaler.fit(df[['energy']])
df['energy'] = scaler.transform(df[['energy']])

scaler.fit(df[['speechiness']])
df['speechiness'] = scaler.transform(df[['speechiness']])

scaler.fit(df[['acousticness']])
df['acousticness'] = scaler.transform(df[['acousticness']])

scaler.fit(df[['instrumentalness']])
df['instrumentalness'] = scaler.transform(df[['instrumentalness']])

scaler.fit(df[['valence']])
df['valence'] = scaler.transform(df[['valence']])

df.head()


# In[5]:


#Inspect Data

#Check updated dataset info
df.info()


# In[6]:


#Inspect Data cont.
from seaborn import pairplot

#Visualize the data in pair plot
pairplot(df)


# In[7]:


#Inspect Data cont.

#Histogram - Popularity rating
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


fig = plt.figure(figsize = (8,8))
ax = fig.gca()
df.hist(ax=ax)
plt.show()


# In[24]:


#Build Model
from sklearn.cluster import KMeans

#Identify best K value using elbow method
k_rng = range(1,10)
sse = []

feature_list = df.drop(["track_name"], axis=1)

#Check the Sum of of squared error
for k in k_rng:
    kM = KMeans(n_clusters = k)
    inertia_value = kM.fit(feature_list)
    sse.append(kM.inertia_)
    print("K-Value = {0}, SSE = {1}".format(i, inertia_value.inertia_))


# In[25]:


plt.figure(figsize=(16,8))
plt.plot(k_rng, sse,'bx-')
plt.xlabel('K')
plt.ylabel('Sum of of squared error')
plt.grid()
plt.show()


# In[9]:


#Train Model and Predict Clusters

#Create model with the best K value determined
km = KMeans(n_clusters = 3)

#Predict Clusters
cluster = km.fit_predict(df[['popularity', 'danceability','energy','speechiness','acousticness',
                                 'instrumentalness','valence']])
cluster


# In[10]:


#Train Model and Predict Clusters cont.

#Add Cluster column to dataset
df['cluster'] = cluster

df


# In[26]:


#Check by manual data input and predict the target
prediction1 = km.predict([[0.750000, 0.848948, 0.944043, 0.378882, 0.007438, 
                           0.003455, 0.820715]])
prediction2 = km.predict([[0.821429, 0.407266, 0.296029, 0.743789, 0.353151, 
                           0.000000, 0.716798]])

print("Original Record Sample Data 1:", prediction1) # target = 0
print("Original Record Sample Data 2:", prediction2) # target = 1


# In[13]:


#Evaluate Model
from sklearn.metrics import silhouette_score

df2 = df.drop(["track_name"], axis=1)

print("Silhouette Score of K-Means:",
silhouette_score(df2, df['cluster'], metric='euclidean', sample_size=None, 
                 random_state=None))


# In[39]:


#Evaluate Model cont.

#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split

features = df2.drop(["cluster"], axis=1)
target = df["cluster"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)

#create Gaussian NB classifier
model = GaussianNB()

#Train the model using training dataset
model.fit(X_train, y_train)

#predict the response for the test dataset
y_pred = model.predict(X_test)



print ("Accuracy: ",metrics. accuracy_score(y_test, y_pred))


# In[32]:


#Evaluate Model cont.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[14]:


#Optimize Algorithm with PCA

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df2 = df.drop(["track_name"], axis=1)

scaled_data = pd.DataFrame(scaler.fit_transform(df2))

scaled_data.head()


# In[15]:


#Before PCA check correlation
import seaborn as sns

sns.heatmap(scaled_data.corr())


# In[42]:


#combine the features
features = ['popularity','danceability','energy','speechiness','acousticness','instrumentalness','valence']
x = df2.loc[:, features].values
y = df2.loc[:, ['cluster']].values


# In[17]:


#Apply PCA to make 2D

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(x)
data_pca = pca.transform(x)
data_pca = pd.DataFrame(data_pca, columns = ['C1', 'C2'])
data_pca.head()


# In[18]:


df2[['cluster']].head()


# In[19]:


#table after PCA(combine new features - PCs with target)
pca_df = pd.concat([data_pca, df2[['cluster']]], axis=1)
pca_df.head()


# In[20]:


pca_df.info()


# In[21]:


#scatter plot to visualize the clusters
fig = plt.figure(figsize = (10,10))
ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)

targets = [0, 1, 2]
colors = ['green', 'pink', 'purple']
for target, color in zip(targets, colors):
    indicesToKeep = pca_df['cluster'] == target
    ax.scatter(pca_df.loc[indicesToKeep, 'C1']
              ,pca_df.loc[indicesToKeep, 'C2']
              , c = color
              , s = 50)

ax.legend(targets)
ax.grid()


# In[23]:


#Evaluate Optimized Model
print("Silhouette Score of PCA:",
silhouette_score(pca_df, pca_df['cluster'], metric='euclidean', sample_size=None, 
                 random_state=None))


# In[33]:


#Evaluate Optimized Model cont.
features = pca_df.drop(["cluster"], axis=1)
target = pca_df["cluster"]

X_train, X_test, y_train, y_test = train_test_split(features, target, 
                                                    test_size=0.3, random_state=0)

#create Gaussian NB classifier
model = GaussianNB()

#Train the model using training dataset
model.fit(X_train, y_train)

#predict the response for the test dataset
y_pred = model.predict (X_test)



print ("Accuracy: ", metrics. accuracy_score(y_test, y_pred))


# In[34]:


#Evaluate Optimized Model cont.
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[ ]:




