import unittest

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.trend_test.trend_test_one_parameter.gumbel_trend_test_one_parameter import GumbelVersusGumbel
from projects.past_extreme_ground_snow_loads.utils import NON_STATIONARY_TREND_TEST_PAPER_1


class TestTrendAnalysis(unittest.TestCase):

    def test_nb_parameters_paper1(self):
        trend_test_classes = NON_STATIONARY_TREND_TEST_PAPER_1
        nb_expected = [2, 3, 3, 4, 3, 4, 4, 5]
        for trend_test_class, nb in zip(trend_test_classes, nb_expected):
            self.assertEqual(trend_test_class.total_number_of_parameters_for_unconstrained_model, nb)

    def test_anderson_goodness_of_fit(self):
        nb_data = 50
        years = list(range(nb_data))
        maxima = GevParams(5, 1, 0).sample(nb_data)
        trend_test = GumbelVersusGumbel(years, maxima, None)
        self.assertTrue(trend_test.goodness_of_fit_anderson_test)


if __name__ == '__main__':
    unittest.main()
