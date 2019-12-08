import numpy as np

index2word = ["I", "Love", "Cheng", "Do"]


def sample_discrete(vec):
    u = np.random.rand()
    start = 0
    for i, num in enumerate(vec):
        if u > start:
            start += num
        else:
            return i - 1
    return i


count = dict([(w, 0) for w in index2word])
# sampling 1000 times
for i in range(1000):
    s = sample_discrete([0.1, 0.5, 0.2, 0.2])
    count[index2word[s]] += 1
# print result
for k in count:
    print("{0}: {1}".format(k, count[k]))
