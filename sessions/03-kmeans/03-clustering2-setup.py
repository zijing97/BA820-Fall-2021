# Learning goals:
## Expand on Distance and now apply Kmeans
## - Kmeans applications
## - Evaluate cluster solutions 
## - hands on with Kmeans and quick review of DBSCan

# resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
# https://scikit-learn.org/stable/modules/clustering.html#dbscan


# installs
# notebook/colab
# ! pip install scikit-plot

# local/server
# pip install scikit-plot



# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt


# COLAB setup ------------------------------------------
# from google.colab import auth
# auth.authenticate_user()
# PROJECT = ''    # <------ change to your project
# SQL = "SELECT * from `questrom.datasets.judges`"
# judges = pd.read_gbq(SQL, PROJECT)


# dataset urls:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MedGPA.csv


# useful code snippets below ---------------------------------

# scale the data
# el_scaler = StandardScaler()
# el_scaler.fit(election)
# election_scaled = el_scaler.transform(election)

# kmeans
# k5 = KMeans(5,  n_init=100)
# judges['k5'] = k5.fit_predict(j)


# k5_centers = k5.cluster_centers_
# sns.scatterplot(data=judges, x="CONT", y="INTG", cmap="virdis", hue="k5")
# plt.scatter(k5_centers[:,0], k5_centers[:,1], c="g", s=100)


# KRANGE = range(2, 30)
# # containers
# ss = []
# for k in KRANGE:
#   km = KMeans(k)
#   lab = km.fit_predict(j)
#   ss.append(km.inertia_)


# skplt.metrics.plot_silhouette(j, k5.predict(j), figsize=(7,7))
ep.index=eo.State
eo.drop(columns='State',inpalce=True)

election = eo.loc[:,'Income':'Dem.Rep']

scaler= StandardScaler()
election_scaled = scaler.fit_transform(election)
type(election_scaled)

hc1=linkage(election_scaled, method='complete')
hc1

# create the plot
plt.figure(figsize=(15,5))
dendrogram(hc1)
plt.show()


#create a clusters
cluster =fcluster(hc1,4,criterion='maxclust')
cluster

eo['cluster']=cluster

#simple profile of a cluster 
eo.groupby('cluster')['Obamawin'].mean()
eo.cluster.value.counts()

eo.loc[eo.cluster]

judges.info()
judges.columns=judges.columns.str.lower()
judges.index=judges.judge
del judges['judge']
judges.dtypes
judges.describe().T

k3=KMeans(3)

k3.fit(judges)
k3_labs=k3.predict(judges)
k3_labs

k3.n_iter_

judges['k3']=k3_labs

judges.sample(3)

#start to ptofile/learn about our cluster
judges.k3.value_counts()



judges.groupby('cluster').mean()

###fit a cluster solution that has 5 clusters
### add to the judges dataset
### how many records is each cluster
### the 'mean' profile to persona s of the 5_cluster solutions

judges.sample(3)
j = judges.copy()
del j['k3']
j.head(3)

judges['k5']=k5.fit_predict(j)
judges.sample(3)

j.shape

k5.cluster_centers_

test_centers = k5.cluster_centers_
test_centers.shape

judges.k5.value_counts()
k5_profile = judges.groupby('k5').mean()
k5_profile

sns.heatmap(k5_profile)
plt.show()

k3.inertia_
k5.inertia_


##exercise

KRANGE= range(2,11)

ss=[]

for k in KRANGE:
    kn = kMean(k)
    lab = kn.fit_predict(j)
    ss.append(kn.intertia_)
sns.lineplot(KRANGE, ss)
plt.show()

k5.inertia_

silo_overall = metrics.silhouette(j,k5.predict(j))
silo_overall

#silo samples

silo_samples=metrics.silhouette_samples(j,k5.predict(j))
type(silo_samples)
silo_samples.shape

skplt.metrics.plot_silhouette(j,k5.predict(j),figsize=(7,7))
plt.show()
