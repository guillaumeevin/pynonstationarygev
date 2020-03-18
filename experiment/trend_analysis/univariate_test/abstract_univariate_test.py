import random
from collections import OrderedDict

import matplotlib.pyplot as plt
from cached_property import cached_property
from matplotlib import colors


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
    NON_SIGNIFICATIVE_TREND = 'non ' + SIGNIFICATIVE + ' trend'

    # this is the most common level of significance
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
    def real_trend_types(cls):
        return [cls.POSITIVE_TREND, cls.NEGATIVE_TREND,
                cls.SIGNIFICATIVE_POSITIVE_TREND, cls.SIGNIFICATIVE_NEGATIVE_TREND, cls.NO_TREND]

    @classmethod
    def three_main_trend_types(cls):
        return [cls.SIGNIFICATIVE_NEGATIVE_TREND, cls.NON_SIGNIFICATIVE_TREND, cls.SIGNIFICATIVE_POSITIVE_TREND]


    @classmethod
    def get_display_trend_type(cls, real_trend_type):
        if cls.SIGNIFICATIVE in real_trend_type:
            return real_trend_type
        else:
            return cls.NON_SIGNIFICATIVE_TREND


    @property
    def time_derivative_of_return_level(self):
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
        assert trend_type in self.real_trend_types(), trend_type
        return trend_type

    @property
    def test_sign(self) -> int:
        raise NotImplementedError

    @property
    def is_significant(self) -> bool:
        raise NotImplementedError





