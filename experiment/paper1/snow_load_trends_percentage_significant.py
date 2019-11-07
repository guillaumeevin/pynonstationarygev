import numpy as np

d = {
    'mu': [8, 4, 1],
    'sigma': [5, 2, 2],
    "both": [9, 6, 5]
}


def compare(a):
    assert a in ['mu', 'sigma']
    percents = [new / old for old, new in zip(d[a], d['both'])]
    print(np.mean(percents))


if __name__ == '__main__':
    compare('mu')
    compare('sigma')
    # conclusion: more than 2 times more significant trends in average
