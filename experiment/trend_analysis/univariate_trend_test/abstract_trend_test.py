import random
import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np

from experiment.trend_analysis.mann_kendall_test import mann_kendall_test
from experiment.trend_analysis.abstract_score import MannKendall


class AbstractTrendTest(object):
    SIGNIFICATIVE = 'significative'
    # 5 possible types of trends
    NO_TREND = 'no trend'
    POSITIVE_TREND = 'positive trend'
    NEGATIVE_TREND = 'negative trend'
    SIGNIFICATIVE_POSITIVE_TREND = SIGNIFICATIVE + ' ' + POSITIVE_TREND
    SIGNIFICATIVE_NEGATIVE_TREND = SIGNIFICATIVE + ' ' + NEGATIVE_TREND

    SIGNIFICANCE_LEVEL = 0.05

    @classmethod
    def trend_type_to_style(cls):
        d = OrderedDict()
        d[cls.POSITIVE_TREND] = 'g--'
        d[cls.NEGATIVE_TREND] = 'r--'
        d[cls.SIGNIFICATIVE_POSITIVE_TREND] = 'g-'
        d[cls.SIGNIFICATIVE_NEGATIVE_TREND] = 'r-'
        d[cls.NO_TREND] = 'k--'
        return d

    @classmethod
    def get_cmap_from_trend_type(cls, trend_type):
        if 'positive' in trend_type:
            return plt.cm.Greens
        elif 'negative' in trend_type:
            return plt.cm.Reds
        else:
            return plt.cm.binary

    def __init__(self, years_after_change_point, maxima_after_change_point):
        self.years_after_change_point = years_after_change_point
        self.maxima_after_change_point = maxima_after_change_point
        assert len(self.years_after_change_point) == len(self.maxima_after_change_point)

    @property
    def n(self):
        return len(self.years_after_change_point)

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


class ExampleRandomTrendTest(AbstractTrendTest):

    @property
    def test_sign(self) -> int:
        return random.randint(0, 2) - 1

    @property
    def is_significant(self) -> bool:
        return random.randint(1, 10) == 10


class MannKendallTrendTest(AbstractTrendTest):

    def __init__(self, years_after_change_point, maxima_after_change_point):
        super().__init__(years_after_change_point, maxima_after_change_point)
        score = MannKendall()
        # Compute score value
        detailed_score = score.get_detailed_score(years_after_change_point, maxima_after_change_point)
        self.score_value = detailed_score[0]
        # Compute the Mann Kendall Test
        MK, S = mann_kendall_test(t=years_after_change_point,
                                  x=maxima_after_change_point,
                                  eps=1e-5,
                                  alpha=self.SIGNIFICANCE_LEVEL,
                                  Ha='upordown')
        assert S == self.score_value
        self.MK = MK

    @property
    def test_sign(self) -> int:
        return np.sign(self.score_value)

    @property
    def is_significant(self) -> bool:
        assert 'reject' in self.MK or 'accept' in self.MK
        return 'accept' in self.MK


class SpearmanRhoTrendTest(AbstractTrendTest):
    pass
