import numpy as np


class AbstractScore(object):

    @classmethod
    def get_detailed_score(cls, sorted_years, sorted_maxima, top_n):
        sorted_maxima = np.array(sorted_maxima)
        year_top_score_max = cls.year_from_top_score(sorted_years[-top_n:], sorted_maxima[-top_n:], top_max=True)
        year_top_score_min = cls.year_from_top_score(sorted_years[:top_n], sorted_maxima[:top_n], top_max=False)
        score_difference = year_top_score_max - year_top_score_min
        return [score_difference, year_top_score_max, year_top_score_min]

    @classmethod
    def year_from_top_score(cls, top_sorted_years, top_sorted_maxima, top_max=None):
        raise NotImplementedError


class MeanScore(AbstractScore):

    @classmethod
    def year_from_top_score(cls, top_sorted_years, top_sorted_maxima, top_max=None):
        return np.mean(top_sorted_years)


class MedianScore(AbstractScore):

    @classmethod
    def year_from_top_score(cls, top_sorted_years, top_sorted_maxima, top_max=None):
        return np.median(top_sorted_years)


class WeigthedScore(AbstractScore):

    @classmethod
    def year_from_top_score(cls, top_sorted_years, top_sorted_maxima, top_max=None):
        assert isinstance(top_max, bool)
        if not top_max:
            top_sorted_maxima = np.sum(top_sorted_maxima) - top_sorted_maxima
        weights = top_sorted_maxima / np.sum(top_sorted_maxima)
        return np.sum(weights * top_sorted_years)
