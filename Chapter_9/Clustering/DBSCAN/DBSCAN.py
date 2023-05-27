#Unlike Kmeans:  DBSCAN canbe uised for non spherical shapes with good denities

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=1000, noise=0.05)

#Tunig epsilon: Observe label differnce between these two epsilons
#Epsioon = 0.2 fits all samples vell in two clusters
dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan = DBSCAN(eps=0.2, min_samples=5)

dbscan.fit(X)

print(f"labels : {dbscan.labels_}")

#DB scan does not implement predict method, to get predict method do teh following
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])