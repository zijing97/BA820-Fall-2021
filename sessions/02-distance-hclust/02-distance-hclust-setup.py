# In this session, we will explore distance calculations and their role in how we can determine 
# similarity between records.  We can use this information to segment our data into like-groups.


# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

# COLAB Only ------------------------------
# from google.colab import auth
# auth.authenticate_user()
# SQL = "SELECT * from `questrom.datasets.mtcars`"
# YOUR_BILLING_PROJECT = ""
# cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

