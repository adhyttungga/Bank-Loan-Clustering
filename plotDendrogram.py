import os
import numpy as np
np.seterr(divide="ignore")
import pandas as pd
import matplotlib.pyplot as plt

path = os.getcwd()
data = pd.read_csv(os.path.join(path,"DS1/app_train.csv"))
print(data.head())

data = data[["INCOME", "APPROVED_CREDIT", "ANNUITY", "PRICE"]]
data = data.head(100)
print(data.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(data)
scaledData = scaler.transform(data)
scaledData = pd.DataFrame(scaledData, columns=data.columns)

from scipy.cluster.hierarchy import dendrogram, ward

linkage_array = ward(scaledData)
dendrogram(linkage_array)

ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [2, 2], "--", c="k")
ax.plot(bounds, [1.6, 1.6], "--", c="k")
ax.text(bounds[1], 2, "two clusters", va="center", fontdict={"size": 15})
ax.text(bounds[1], 1.6, "four clusters", va="center", fontdict={"size": 15})
plt.xlabel("Sample index")
plt.ylabel("Cluster distance")

plt.show()