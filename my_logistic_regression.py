import numpy as np

class MyLogisticRegression():

    def __init__(self, theta, alpha=1e-4, max_iter=10000):
        if isinstance(theta, (list, np.ndarray)) is False or isinstance(alpha, float) is False or isinstance(max_iter, int) is False:
            return None
        self.alpha = alpha
        self.max_iter = max_iter
        if isinstance(theta, list):
            self.thetas = np.array(theta)
        else:
            self.thetas = theta
        self.thetas = self.thetas.astype('float64')

    def predict_(self, x):
        if isinstance(x, np.ndarray) is False:
            return None
        if x.size == 0 or self.thetas.size == 0:
            return None
        if x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        return 1 / (1 + np.exp(-np.dot(np.c_[np.ones(x.shape[0]), x], self.thetas)))

    def loss_(self, y, y_hat):
        if isinstance(y, np.ndarray) is False or isinstance(y_hat, np.ndarray) is False:
            return None
        if y.size == 0 or y_hat.size == 0:
            return None
        if y.shape[0] != y_hat.shape[0]:
            return None
        eps = 1e-3
        y = y.astype('float64')
        y_hat = y_hat.astype('float64')
        for item in y_hat:
            if item[0] == 0:
                item[0] = item[0] + eps
            elif item[0] == 1:
                item[0] = item[0] - eps
        return np.sum((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))) / -len(y)

    def fit_(self, x, y):
        if isinstance(x, np.ndarray) is False or isinstance(y, np.ndarray) is False:
            return None
        if x.size == 0 or y.size == 0 or self.thetas.size == 0:
            return None
        if x.shape[1] + 1 != self.thetas.shape[0] or y.shape[1] != 1 or self.thetas.shape[1] != 1:
            return None

        gradient = []
        for epoch in range(self.max_iter + 1):
            gradient = np.asarray(np.matmul(np.transpose(np.c_[np.ones(x.shape[0]), x]), (self.predict_(x) - y)) / len(x)).astype('float64')
            self.thetas = self.thetas - (self.alpha * gradient)
            if epoch % 10000 == 0:
                print('{:05d}: loss: {}'.format(epoch, self.loss_(self.predict_(x), y)))
        return self.thetas
