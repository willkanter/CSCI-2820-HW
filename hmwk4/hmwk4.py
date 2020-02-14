import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from numpy import linalg as LA

df = pd.read_csv(sys.argv[1], header=None) # reading

# print(df.head(15)) # this prints the first 15 rows of our columns



k = int(sys.argv[2]) # set k equal to the second command line argument

kGrouping = [-1 for x in range(len(df))]
# create an array that will correspond to df's index with it's k grouping (0-k)
npa = df.to_numpy()

centersX = df[0].sample(n = k)
centersY = df[1].sample(n = k)
centers = np.array(list(zip(centersX, centersY)))

def kMeans(k, df):
    for i in range(len(df)):
        kGrouping[i] = np.argmin([LA.norm(npa[i]-centers[j]) for j in range(k)]) # magic python magic
        # print(kGrouping[i])




kMeans(k, df)

sizeOfGroup = [0 for x in range(k)]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
for i in range(k):
    # print(kGrouping)
    pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
    # print(pointsInGroup)
    plt.scatter(pointsInGroup[:,0], pointsInGroup[:,1], c=colors[i])

# print(centers)
plt.scatter(centers[:, 0], centers[:, 1], c='k')
plt.show()

print("These are the centers for first iteration: ")
print(centers)


def updateCenters(x):
    for i in range(k):
        pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
        x[i] = [np.mean(pointsInGroup[:, 0]), np.mean(pointsInGroup[:, 1])]

# print(kGrouping)
# print(centers)

for i in range(100):
    updateCenters(centers)
    kMeans(k, df)
    # print(kGrouping)
    if i is 50:
        print("Halfway through iterations")
        print(centers)
    # plt.scatter(centers[:, 0], centers[:, 1], c='k')

# print(pointsInGroup)
print("New centers after 100 iterations")
print(centers)



for i in range(k):
    # print(kGrouping)
    pointsInGroup = np.array([npa[j] for j in range(len(df)) if kGrouping[j] == i])
    # print(pointsInGroup)
    plt.scatter(pointsInGroup[:, 0], pointsInGroup[:, 1], c=colors[i])

plt.scatter(centers[:, 0], centers[:, 1], c='k')
plt.show()
