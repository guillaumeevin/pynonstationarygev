import random
import warnings

import matplotlib.pyplot as plt
from collections import OrderedDict

import numpy as np
from cached_property import cached_property

from experiment.trend_analysis.mann_kendall_test import mann_kendall_test
from experiment.trend_analysis.abstract_score import MannKendall
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest


class MannKendallTrendTest(AbstractUnivariateTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year)
        score = MannKendall()
        # Compute score value
        detailed_score = score.get_detailed_score(self.years_after_starting_year, self.maxima_after_starting_year)
        self.score_value = detailed_score[0]
        # Compute the Mann Kendall Test
        MK, S = mann_kendall_test(t=self.years_after_starting_year,
                                  x=self.maxima_after_starting_year,
                                  eps=1e-5,
                                  alpha=self.SIGNIFICANCE_LEVEL,
                                  Ha='upordown')
        # Raise warning if scores are differents
        if S != self.score_value:
            warnings.warn('S={} is different that score_value={}'.format(S, self.score_value), WarningScoreValue)
        self.MK = MK

    @property
    def test_sign(self) -> int:
        return np.sign(self.score_value)

    @property
    def is_significant(self) -> bool:
        assert 'reject' in self.MK or 'accept' in self.MK
        return 'accept' in self.MK


class SpearmanRhoTrendTest(AbstractUnivariateTest):
    pass
