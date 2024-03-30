import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib.dates import date2num

df = pd.read_csv("./NFLX.csv")

df = df.iloc[100:,:]
df['Date']=pd.to_datetime(df['Date'])
# df['Date']=df['Date'].values.astype("float64")
fig, ax = plt.subplots(figsize=(20,8))
plt.plot(df["Date"], df["Open"], color='Red')
ax.set_xlabel('Date', fontsize='11')
ax.set_ylabel('Opening Price in USD', fontsize='11')
plt.title('Netflix Stock Prices (April 13th 2018 ~)')
plt.grid()
plt.show()

df = df.drop(columns=['Low','Close','Volume','Adj Close'])
for i in range(len(df)):
    df['Date'][i+100] = i
print(df)
# print((df['Date']))
# create model
nbrs = NearestNeighbors(n_neighbors = 10)
# fit model
nbrs.fit(df)
# distances and indexes of k-neighbors from model outputs
distances, indexes = nbrs.kneighbors(df)
# plot
plt.figure(figsize=(15, 7))
plt.plot(distances.mean(axis =1))
plt.show()
distances = pd.DataFrame(distances)
distances_mean = distances.mean(axis =1)
print(distances_mean)
distances_mean.describe()
th = 9.0
outlier_index = np.where(distances_mean > th)
print(outlier_index)
outlier_values = df.iloc[outlier_index]
print(outlier_values)
# plot data
plt.figure(figsize=(20, 7))
plt.plot(df["Date"], df["Open"], color = "b")
# plot outlier values
plt.scatter(outlier_values["Date"], outlier_values["Open"], color = "r")
plt.show()