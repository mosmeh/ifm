#!/usr/bin/python3

import numpy as np

data = np.load('positive-data.npy')

class IFM(object):
    def __init__(self, d, k=40, eta=0.004, lambda_0=2.0, lambda_w=8.0, lambda_V=None):
        self.d = d
        self.k = k
        self.eta = eta

        self.lambda_0 = lambda_0
        self.lambda_w = lambda_w
        self.lambda_V = np.ones(k) * 16.0 if lambda_V is None else lambda_V

        self.w_0 = 0
        self.w = np.zeros(d)
        self.V = np.random.normal(0, 0.01, (d, k))

    def predict(self, x):
        return self.w_0 + np.dot(self.w, x) + np.sum(np.triu(np.dot(self.V, self.V.T)) * np.outer(x, x))

    def loss(self, x, target):
        return (self.predict(x) - target) ** 2 / 2

    def train(self, x, target):
        # lambda update
        self.lambda_0 = max(0, self.lambda_0 - self.eta * self.w_0 * self.w_0)
        self.lambda_w = max(0, self.lambda_w - self.eta * np.dot(self.w, self.w))
        self.lambda_V = np.maximum(0, self.lambda_V - self.eta * np.sum(self.V ** 2, axis=0))

        # theta update
        residual = self.predict(x) - target
        print('loss: %f' % (residual ** 2 / 2))

        self.w_0 -= self.eta * (residual + 2 * self.lambda_0 * self.w_0)
        s = np.sum(x * self.V.T, axis=1)
        for i in range(self.d):
            if x[i] != 0:
                self.w[i] -= self.eta * (residual * x[i] + 2 * self.lambda_w * self.w[i])
                self.V[i] -= self.eta * (residual * x[i] * s + 2 * self.lambda_V * self.V[i])

ifm = IFM(d=len(data[0]), eta=0.0001)
n_samples = len(data)
b = int(n_samples * 0.3)
for (i, x) in enumerate(data[:b], start=1):
    print('%i / %i' % (i, b))
    ifm.train(x, 1)

import pickle
with open('b-ifm.dump', 'wb') as f:
    pickle.dump(ifm, f)

#for (i, x) in enumerate(data[b:], start=1)
    #print('%i / %i' % (i, n_samples - b))
