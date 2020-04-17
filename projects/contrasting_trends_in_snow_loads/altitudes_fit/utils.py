import operator
from enum import Enum


class Score(Enum):
    NLLH_TEST = 1
    AIC_TRAIN = 2
    BIC_TRAIN = 3
    SPAN_30_RETURN_LEVEL_2019_AT_1000M = 4


class Grouping(Enum):
    MEAN_RANKING = 1


def get_key_with_max_value(d):
    return max(d.items(), key=operator.itemgetter(1))[0]


def get_key_with_min_value(d):
    return min(d.items(), key=operator.itemgetter(1))[0]
