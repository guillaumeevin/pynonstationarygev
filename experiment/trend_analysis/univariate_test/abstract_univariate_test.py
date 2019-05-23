import random
import warnings

import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from experiment.trend_analysis.mann_kendall_test import mann_kendall_test
from experiment.trend_analysis.abstract_score import MannKendall


class AbstractUnivariateTest(object):
    SIGNIFICATIVE = 'significative'
    # 5 possible types of trends
    NO_TREND = 'no trend'
    ALL_TREND = 'all trend'
    POSITIVE_TREND = 'positive trend'
    NEGATIVE_TREND = 'negative trend'
    SIGNIFICATIVE_ALL_TREND = SIGNIFICATIVE + ' ' + ALL_TREND
    SIGNIFICATIVE_POSITIVE_TREND = SIGNIFICATIVE + ' ' + POSITIVE_TREND
    SIGNIFICATIVE_NEGATIVE_TREND = SIGNIFICATIVE + ' ' + NEGATIVE_TREND

    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, years, maxima, starting_year):
        self.years = years
        self.maxima = maxima
        self.starting_year = starting_year
        assert len(self.years) == len(self.maxima)

    @cached_property
    def idx_for_starting_year(self):
        return self.years.index(self.starting_year)

    @property
    def years_after_starting_year(self):
        return self.years[self.idx_for_starting_year:]

    @property
    def maxima_after_starting_year(self):
        return self.maxima[self.idx_for_starting_year:]

    @classmethod
    def trend_type_to_style(cls):
        d = OrderedDict()
        d[cls.ALL_TREND] = 'k--'
        d[cls.POSITIVE_TREND] = 'g--'
        d[cls.NEGATIVE_TREND] = 'r--'
        d[cls.SIGNIFICATIVE_ALL_TREND] = 'k-'
        d[cls.SIGNIFICATIVE_POSITIVE_TREND] = 'g-'
        d[cls.SIGNIFICATIVE_NEGATIVE_TREND] = 'r-'
        # d[cls.NO_TREND] = 'k--'
        return d

    @classmethod
    def get_trend_types(cls, trend_type):
        if trend_type is cls.ALL_TREND:
            return cls.get_trend_types(cls.POSITIVE_TREND) + cls.get_trend_types(cls.NEGATIVE_TREND)
        elif trend_type is cls.SIGNIFICATIVE_ALL_TREND:
            return [cls.SIGNIFICATIVE_POSITIVE_TREND, cls.SIGNIFICATIVE_NEGATIVE_TREND]
        if trend_type is cls.POSITIVE_TREND:
            return [cls.POSITIVE_TREND, cls.SIGNIFICATIVE_POSITIVE_TREND]
        elif trend_type is cls.NEGATIVE_TREND:
            return [cls.NEGATIVE_TREND, cls.SIGNIFICATIVE_NEGATIVE_TREND]
        else:
            return [trend_type]

    @classmethod
    def get_cmap_from_trend_type(cls, trend_type):
        if 'positive' in trend_type:
            return plt.cm.Greens
        elif 'negative' in trend_type:
            return plt.cm.Reds
        else:
            return plt.cm.binary

    @property
    def n(self):
        return len(self.years)

    @property
    def test_trend_strength(self):
        return 0.0

    @property
    def test_trend_type(self) -> str:
        test_sign = self.test_sign
        assert test_sign in [-1, 0, 1]
        if test_sign == 0:
            trend_type = self.NO_TREND
        else:
            trend_type = self.POSITIVE_TREND if test_sign > 0 else self.NEGATIVE_TREND
            if self.is_significant:
                trend_type = self.SIGNIFICATIVE + ' ' + trend_type
        assert trend_type in self.trend_type_to_style()
        return trend_type

    @property
    def test_sign(self) -> int:
        raise NotImplementedError

    @property
    def is_significant(self) -> bool:
        raise NotImplementedError


class ExampleRandomTrendTest(AbstractUnivariateTest):

    @property
    def test_sign(self) -> int:
        return random.randint(0, 2) - 1

    @property
    def is_significant(self) -> bool:
        return random.randint(1, 10) == 10


class WarningScoreValue(Warning):
    pass
