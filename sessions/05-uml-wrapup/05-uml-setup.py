######################################################
## Wrap up UML
## Hands on-heavy class to dive into some concepts you may want to explore 
## Learning objectives:
##
## 0. wrap up PCA
## 1. exposure to more contemporary techniques for interviews, awareness, and further exploration
## 2. highlight different use-cases, some that help with viz for non-tech, others to think about alternatives to linear PCA
## 3. use this as a jumping off point for you to consider the fact that are lots of methods, and its not typically for this task, do this one approach
######################################################

# installs

# notebook install
# ! pip install umap-learn
# local
# pip install umap-learn

# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# scipy
from scipy.spatial.distance import pdist

# scikit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans


