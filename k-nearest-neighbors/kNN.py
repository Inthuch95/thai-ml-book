# k-Nearest Neighbors
import numpy as np

def kNN(X_train, Y_train, X_test, k):
	Y_test = []
	for x in X_test:
		# Euclidean distance
		d = np.sqrt(np.sum((X_train - x) **2, axis=1))
		idx = np.argsort(d)
		(value, counts) = np.unique(Y_train[idx[:k]], return_counts=True)
		ind = np.argmax(counts)
		Y_test.append(values[ind])
	return Y_test