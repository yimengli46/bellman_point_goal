from random import uniform as rand
import random


def ranks(sample):
    """
    Return the ranks of each element in an integer sample.
    """
    indices = sorted(range(len(sample)), key=lambda i: sample[i])
    return sorted(indices, key=lambda i: indices[i])


def sample_with_minimum_distance(k=3, d=2):
    """
    Sample of k elements from range(n), with a minimum distance d.
    """

    sample = random.sample(range(-9, 9), k)
    return [s + (d - 1) * r for s, r in zip(sample, ranks(sample))]


def randConstrained(env, n, M):
    random.seed(env)
    splits = [0] + [rand(0, 1) for _ in range(0, n - 1)] + [1]
    splits.sort()
    diffs = [x - splits[i - 1] for i, x in enumerate(splits)][1:]
    result = map(lambda x: x * M, diffs)
    return result
