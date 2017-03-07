#!/usr/bin/python3

import numpy as np
import datetime
from collections import defaultdict

occupations = list(np.loadtxt('u.occupation', dtype=bytes))
converters = {
    2: lambda s: 0 if s == b'M' else 1,
    3: lambda s: occupations.index(s)
}
demographics = np.loadtxt('u.user', delimiter='|', usecols=(0, 1, 2, 3), converters=converters, dtype=int)
demographics_dict = dict(zip(demographics[:, 0], demographics[:, 1:]))

item_info = np.loadtxt('u.item.utf-8', delimiter='|', usecols=tuple([0] + list(range(5, 24))), dtype=int)
item_info_dict = dict(zip(item_info[:, 0], item_info[:, 1:]))

genres = np.loadtxt('u.genre', delimiter='|', usecols=0, dtype=bytes)

ratings = np.loadtxt('u.data', dtype=int)

num_users = len(demographics)
num_items = len(item_info)
num_genres = len(genres)
num_features = num_users + len(demographics[0]) - 1 + num_items + num_genres * 2 + 7 * 2

def to_one_hot(i, size):
    a = np.zeros(size, dtype=int)
    a[i] = 1
    return a

last_rated_genre = defaultdict(lambda: np.zeros(num_genres))
last_rated_day = defaultdict(lambda: None)
data = np.ndarray((len(ratings), num_features), dtype=int)
for (i, (user_id, item_id, rating, timestamp)) in enumerate(sorted(ratings, key=lambda x: x[3])):
    weekday = datetime.datetime.fromtimestamp(timestamp).weekday()
    data[i] = np.hstack([to_one_hot(user_id - 1, num_users),
                         demographics_dict[user_id],
                         to_one_hot(item_id - 1, num_items),
                         item_info_dict[item_id],
                         last_rated_genre[user_id],
                         to_one_hot(weekday, 7),
                         to_one_hot(last_rated_day[user_id], 7)])
    last_rated_genre[user_id] = item_info_dict[item_id]
    last_rated_day[user_id] = weekday

np.save('data.npy', data)
np.save('targets.npy', ratings[:, 2])
