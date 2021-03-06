import numpy as np
import matplotlib.pyplot as plt
np.random.seed(24)
class GMM(object):
    """Gaussian Mixture Model
    """

    def __init__(self, data, K):
        """
        K: the number of gaussian models
        alpha: the weight for corresponding gaussian model
        mu: the vector of means
        sigma2: the vector of variances
        N: the number of samples
        K: the number of gaussian models
        """

        self.data = data
        self.K = K
        self.alpha = (np.ones(K) / K)  # shape (1,K)
        self.mu = np.linspace(data.min(), data.max(), num=K)  # shape (1,K)
        self.sigma2 = (np.ones(K))  # shape(1,K)
        self.N = len(data)
        self.gamma = np.ones((self.N, K)) / K  # shape(1,K)

    def norm(self):
        # shape(K, N)
        gaussian = (1 / np.sqrt(2 * np.pi * self.sigma2.reshape(self.K, 1)) * np.exp(
            - (self.data - self.mu.reshape(self.K, 1)) ** 2 / (2 * self.sigma2.reshape(self.K, 1))))
        return gaussian

    def fit(self):
        sigma2_ = self.sigma2
        mu_ = self.mu
        epoch = 0

        while True:
            print("epoch:", epoch)
            epoch += 1
            # gamma.shape(N, K)
            self.gamma = self.norm().T * self.alpha / (self.norm().T * self.alpha).sum(axis=1).reshape(self.N, 1)
            # mu.shape(1, K)
            self.mu = np.matmul(self.data, self.gamma) / self.gamma.sum(axis=0)
            # sigma2.shape(1,K)
            self.sigma2 = (self.gamma * (data.reshape(self.N, 1) - self.mu) ** 2).sum(axis=0) / self.gamma.sum(axis=0)
            # alpha.shape(1, K)
            self.alpha = self.gamma.sum(axis=0) / self.N
            # print("a",self.alpha)
            # print("mu",self.mu)
            # print("g",self.gamma)
            if (np.sum((self.mu - mu_) ** 2) + np.abs(self.sigma2 - sigma2_).sum()) < 1e-6:
                break

            mu_ = self.mu
            sigma2_ = self.sigma2
            #print(self.gamma.argmax(axis=1))

        return self.gamma.argmax(axis=1)


data = np.concatenate((np.random.normal(-4, 1, 33), np.random.normal(4, 1, 34), np.random.normal(12, 1, 33)))
gmm = GMM(data, 3)
label = gmm.fit()
plt.figure()
plt.scatter(data,np.zeros(shape=100),c=label)
plt.show()
