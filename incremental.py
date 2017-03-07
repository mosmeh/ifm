#!/usr/bin/python3

import pickle
import datetime
import copy
import random
import numpy as np
from collections import defaultdict
from progressbar import ProgressBar
import multiprocessing as mp

ETA = 0.0001
class IFM(object):
    def __init__(self, d, k=40, lambda_0=2.0, lambda_w=8.0, lambda_V=None):
        self.d = d
        self.k = k

        self.lambda_0 = lambda_0
        self.lambda_w = lambda_w
        self.lambda_V = np.ones(k) * 16.0 if lambda_V is None else lambda_V

        self.w_0 = 0
        self.w = np.zeros(d)
        self.V = np.random.normal(0, 0.0001, (d, k))
        self.backup_params()

    def predict(self, x):
        res = self.w_0 + np.dot(self.w, x)
        for i in range(self.d):
            if x[i] != 0:
                for j in range(i, self.d):
                    if x[j] != 0:
                        res += np.dot(self.V[i],  self.V[j]) * x[i] * x[j]
        return res

    def loss(self, x, target):
        return (self.predict(x) - target) ** 2

    def backup_params(self):
        self.prev_w_0 = self.w_0
        self.prev_w = self.w
        self.prev_V = self.V

    def train(self, x, target, eta=ETA):
        residual = 2 * (self.predict(x) - target)
        s = np.sum(x * self.V.T, axis=1)
        prev_s = np.sum(x * self.prev_V.T, axis=1)

        #lambda update
        self.lambda_0 = max(0,        self.lambda_0 - eta * residual * -2 * eta * self.prev_w_0)
        self.lambda_w = max(0,        self.lambda_w - eta * residual * -2 * eta * np.dot(self.prev_w, x))
        self.lambda_V = np.maximum(0, self.lambda_V - eta * residual * -2 * eta * (s * prev_s - np.sum((x * x) * (self.V * self.prev_V).T, axis=1)))

        # theta update
        self.backup_params()

        self.w_0 -= eta * (residual + 2 * self.lambda_0 * self.w_0)
        for i in range(self.d):
            if x[i] != 0:
                self.w[i] -= eta * (residual * x[i] + 2 * self.lambda_w * self.w[i])
                self.V[i] -= eta * (residual * x[i] * s + 2 * self.lambda_V * self.V[i])

    def batch_train2(self, xs, targets, xsd, targetsd, eta=ETA):
        valid_set = list(zip(xsd, targetsd))
        for (x, target) in zip(xs, targets):
            residual = 2 * (self.predict(x) - target)
            s = np.sum(x * self.V.T, axis=1)
            prev_s = np.sum(x * self.prev_V.T, axis=1)

            # theta update
            self.backup_params()

            self.w_0 -= eta * (residual + 2 * self.lambda_0 * self.w_0)
            for i in range(self.d):
                if x[i] != 0:
                    self.w[i] -= eta * (residual * x[i] + 2 * self.lambda_w * self.w[i])
                    self.V[i] -= eta * (residual * x[i] * s + 2 * self.lambda_V * self.V[i])

            #lambda update
            valid_x, valid_target = random.sample(valid_set, 1)[0]
            valid_residual = 2 * (self.predict(valid_x) - valid_target)
            self.lambda_0 = max(0,        self.lambda_0 - eta * valid_residual * -2 * eta * self.prev_w_0)
            self.lambda_w = max(0,        self.lambda_w - eta * valid_residual * -2 * eta * np.dot(self.prev_w, valid_x))
            self.lambda_V = np.maximum(0, self.lambda_V - eta * valid_residual * -2 * eta * (s * prev_s - np.sum((valid_x * valid_x) * (self.V * self.prev_V).T, axis=1)))


    def update_lambda(self, xs, targets, eta=ETA):
        dl0 = 0
        dlw = 0
        dlV = np.zeros_like(self.lambda_V)

        for (x, target) in zip(xs, targets):
            residual = 2 * (self.predict(x) - target)
            s = np.sum(x * self.V.T, axis=1)
            prev_s = np.sum(x * self.prev_V.T, axis=1)

            dl0 += residual * -2 * eta * self.prev_w_0
            dlw += residual * -2 * eta * np.dot(self.prev_w, x)
            dlV += residual * -2 * eta * (s * prev_s - np.sum(x * x * (self.V * self.prev_V).T, axis=1))

        self.lambda_0 = max(0,        self.lambda_0 - eta * dl0 / len(xs))
        self.lambda_w = max(0,        self.lambda_w - eta * dlw / len(xs))
        self.lambda_V = np.maximum(0, self.lambda_V - eta * dlV / len(xs))

    def batch_train(self, xs, targets, eta=ETA):
        d0 = 0
        dw = np.zeros_like(self.w)
        dV = np.zeros_like(self.V)

        for (x, target) in zip(xs, targets):
            residual = 2 * (self.predict(x) - target)

            d0 += residual + 2 * self.lambda_0 * self.w_0
            s = np.sum(x * self.V.T, axis=1)
            for i in range(self.d):
                if x[i] != 0:
                    dw += residual * x[i] + 2 * self.lambda_w * self.w[i]
                    dV += residual * x[i] * s + 2 * self.lambda_V * self.V[i]

        self.backup_params()
        self.w_0 -= eta * d0 / len(xs)
        self.w   -= eta * dw / len(xs)
        self.V   -= eta * dV / len(xs)

def to_one_hot(i, size):
    a = np.zeros(size, dtype=int)
    if i is None:
        return a
    a[i] = 1
    return a

occupations = list(np.loadtxt('u.occupation', dtype=bytes))
converters = {
    2: lambda s: 0 if s == b'M' else 1,
    3: lambda s: occupations.index(s)
}
demographics = np.loadtxt('u.user', delimiter='|', usecols=(0, 1, 2, 3), converters=converters, dtype=int)
demographics_dict = dict(zip(demographics[:, 0], map(lambda x: np.hstack([x[0:2], to_one_hot(x[2], len(occupations))]),demographics[:, 1:])))

item_info = np.loadtxt('u.item.utf-8', delimiter='|', usecols=tuple([0] + list(range(6, 24))), dtype=int)
item_info_dict = dict(zip(item_info[:, 0], item_info[:, 1:]))

data = np.loadtxt('u.data', dtype=int)
data = sorted(filter(lambda x: x[2] >= 5, data), key=lambda x: x[3])

user_ids = list(set(map(lambda x: x[0], data)))
item_ids = list(set(map(lambda x: x[1], data)))

num_users = len(user_ids)
num_items = len(item_ids)
num_genres = len(item_info_dict[item_ids[0]])
num_features = num_users + len(demographics_dict[user_ids[0]]) + num_items + num_genres * 2 + 7 * 2
print('features: %i' % num_features)

ifm = IFM(d=num_features, k=40)

n_samples = len(data)
batch_train_size = int(n_samples * 0.2)
validation_size = int(n_samples * 0.1)
batch_size = batch_train_size + validation_size

last_rated_genre = defaultdict(lambda: np.zeros(num_genres))
last_rated_day = defaultdict(lambda: None)

batch_data = np.ndarray((batch_size, num_features), dtype=int)
for (i, (user_id, item_id, rating, timestamp)) in enumerate(data[:batch_size]):
    weekday = datetime.date.fromtimestamp(timestamp).weekday()
    batch_data[i] = np.hstack([to_one_hot(user_ids.index(user_id), num_users),
                               demographics_dict[user_id],
                               to_one_hot(item_ids.index(item_id), num_items),
                               item_info_dict[item_id],
                               last_rated_genre[user_id],
                               to_one_hot(weekday, 7),
                               to_one_hot(last_rated_day[user_id], 7)])
    last_rated_genre[user_id] = item_info_dict[item_id]
    last_rated_day[user_id] = weekday

batch_train_data = batch_data[:batch_train_size]
validation_data = batch_data[batch_train_size:]

prev_loss = np.inf
prev_ifm = None
while True:
    np.random.shuffle(batch_train_data)
    #ifm.batch_train(batch_train_data, np.ones(batch_train_size))
    #ifm.update_lambda(validation_data, np.ones(validation_size))
    ifm.batch_train2(batch_train_data, np.ones(batch_train_size), validation_data, np.ones(validation_size))

    loss = 0
    for x in validation_data:
        loss += ifm.loss(x, 1)

    print('loss: %f' % loss)
    if loss >= prev_loss:
        ifm = copy.deepcopy(prev_ifm)
        break
    elif prev_loss - loss < 1:
        break
    else:
        prev_loss = loss
        prev_ifm = copy.deepcopy(ifm)

with open('ifm-l.dump', 'wb') as f:
    pickle.dump(ifm, f)
#ifm = pickle.load(open('ifm-l.dump', 'rb'))

for x in validation_data:
    ifm.train(x, 1)

n = 10
t = 3000
recalls = []
recalls_static = []

pool = mp.Pool(4)
static_fm = copy.deepcopy(ifm)
with open('recalls-l.log', 'w') as f:
    for (i, (user_id, item_id, rating, timestamp)) in enumerate(data[batch_size:], start=1):
        weekday = datetime.date.fromtimestamp(timestamp).weekday()

        user_info = np.hstack([to_one_hot(user_ids.index(user_id), num_users),
                               demographics_dict[user_id]])
        contextual_info = np.hstack([last_rated_genre[user_id],
                                     to_one_hot(weekday, 7),
                                     to_one_hot(last_rated_day[user_id], 7)])

        features = [np.hstack([user_info,
                               to_one_hot(item_ids.index(iid), num_items),
                               item_info_dict[iid],
                               contextual_info]) for iid in item_ids]

        predictions = list(map(lambda x: abs(x - 1), pool.map(ifm.predict, features)))
        l = list(map(lambda x: x[1], sorted(zip(predictions, item_ids))[:n]))
        recall = 1 if item_id in l else 0
        recalls.append(recall)
        moving_avg = np.average(recalls[-t:])

        predictions_static = list(map(lambda x: abs(x - 1), pool.map(static_fm.predict, features)))
        l_static = list(map(lambda x: x[1], sorted(zip(predictions_static, item_ids))[:n]))
        recall_static = 1 if item_id in l_static else 0
        recalls_static.append(recall_static)
        moving_avg_static = np.average(recalls_static[-t:])

        print('%i/%i:' % (i, n_samples - batch_size))
        print('incremental: top pred.:%f recall:%i moving avg.:%f' % (np.min(predictions), recall, moving_avg))
        print('static:      top pred.:%f recall:%i moving avg.:%f' % (np.min(predictions_static), recall_static, moving_avg_static))
        f.write('%f %f\n' % (moving_avg, moving_avg_static))
        #f.write('%f\n' % moving_avg)
        f.flush()

        x = np.hstack([user_info,
                       to_one_hot(item_ids.index(item_id), num_items),
                       item_info_dict[item_id],
                       contextual_info])
        last_rated_genre[user_id] = item_info_dict[item_id]
        last_rated_day[user_id] = weekday
        ifm.train(x, 1)

        if i % 1000 == 0:
            with open('ifm-stream-l-%i.dump' % i, 'wb') as fi:
                pickle.dump(ifm, fi)
