import numpy as np
from collections import Counter

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def euclidean_distance(point1, point2):

    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimension")
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

class KNN:
    def __init__(self, k=3, task = "classification"):
        self.k, self.task = k, task

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if self.task == "classification":
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        
        elif self.task == "regression":
            return np.mean(k_nearest_labels)
        
    def is_anomaly(self, x , threshold):
        distances = [euclidean_distance(x, x_train) for x_train in X_train]
        k_nearst = np.sort(distances)[:self.k]
        return k_nearst > threshold * threshold

        

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y) # Num of correct predictions / total actual values
        return accuracy
    
if __name__ == "__main__":

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")