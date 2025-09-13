import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

 #It’s the proportion of variance in the dependent variable (y_true)
# that is predictable from the independent variable (y_pred).
def r2_score(y_true, y_pred):
    # corr is the Pearson correlation coefficient between y_true and y_pred.
    corr_matrix = np.corrcoef(y_true, y_pred)
    # Extracts the off-diagonal value (the correlation between the two arrays).
    corr = corr_matrix[0, 1]
    return corr ** 2

def mean_squared_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

class LinearRegression:
    def __init__(self, learning_rate=0.001, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, Y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Decent
        for _ in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - Y))
            self.weights -= self.learning_rate * dw

            db = (1/n_samples) * np.sum(y_predicted - Y)
            self.bias -= self.learning_rate * db


    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted

    


if __name__ == "__main__":

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=45
    )

    regressor = LinearRegression(learning_rate=0.01, num_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    print("MSE:", mse)

    accu = r2_score(y_test, predictions)
    print("Accuracy:", accu)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()



