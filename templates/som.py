import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np

data=pd.read_csv("static\pd_speech_features.csv")
data.head()



col_names = data.columns
for c in col_names:
    data[c] = data[c].replace("?", np.NaN)

data = data.apply(lambda x:x.fillna(x.value_counts().index[0]))


X = data.values[:, 0:754]
Y = data.values[:,754]




net = sps.somNet(25,25, X, PBC=True)
net.train(0.01, 20)

net.diff_graph(show=True)
