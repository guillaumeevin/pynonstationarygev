import random


class AbstractTrendTest(object):
    SIGNIFICATIVE = 'significative'
    # 5 possible types of trends
    NO_TREND = 'no trend'
    POSITIVE_TREND = 'positive trend'
    NEGATIVE_TREND = 'negative trend'
    SIGNIFICATIVE_POSITIVE_TREND = SIGNIFICATIVE + ' ' + POSITIVE_TREND
    SIGNIFICATIVE_NEGATIVE_TREND = SIGNIFICATIVE + ' ' + NEGATIVE_TREND

    # todo: maybe create ordered dict
    TREND_TYPE_TO_STYLE = {
        NO_TREND: 'k--',
        POSITIVE_TREND: 'g--',
        SIGNIFICATIVE_POSITIVE_TREND: 'g-',
        SIGNIFICATIVE_NEGATIVE_TREND: 'r-',
        NEGATIVE_TREND: 'r--',
    }

    TREND_TYPES = list(TREND_TYPE_TO_STYLE.keys())

    def __init__(self, years_after_change_point, maxima_after_change_point):
        self.years_after_change_point = years_after_change_point
        self.maxima_after_change_point = maxima_after_change_point

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
        assert trend_type in self.TREND_TYPE_TO_STYLE
        return trend_type

    @property
    def test_sign(self) -> int:
        raise NotImplementedError

    @property
    def is_significant(self) -> bool:
        raise NotImplementedError


class ExampleTrendTest(AbstractTrendTest):

    @property
    def test_sign(self) -> int:
        return random.randint(0, 2) - 1

    @property
    def is_significant(self) -> bool:
        return random.randint(1, 10) == 10


class MannKendallTrendTest(AbstractTrendTest):
    pass
