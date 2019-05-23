import numpy as np


class AbstractTrendScore(object):
    """A score that should be equal to zero is there is no trend
    positive if we suppose a positive trend
    negative if we suppose a negative trend

    We don't care what happen before the change point.
    All we want to focus on, is the potential trend that could exist in the data after a potential change point"""

    def __init__(self, number_of_top_values=None) -> None:
        self.number_of_top_values = number_of_top_values

    def get_detailed_score(self, years_after_change_point, maxima_after_change_point):
        raise NotImplementedError


class MannKendall(AbstractTrendScore):
    # see here for the explanation: https://up-rs-esp.github.io/mkt/

    def get_detailed_score(self, years_after_change_point, maxima_after_change_point):
        score = 0.0
        for i, xi in enumerate(maxima_after_change_point[:-1]):
            for xj in maxima_after_change_point[i + 1:]:
                score += np.sign(xj - xi)
        return [score, score, score]


class SortedScore(AbstractTrendScore):

    def __init__(self, number_of_top_values=None) -> None:
        super().__init__(number_of_top_values)
        assert self.number_of_top_values is not None

    def get_detailed_score(self, years_after_change_point, maxima_after_change_point):
        # Get sorted years and sorted maxima
        sorted_years, sorted_maxima = zip(
            *sorted(zip(years_after_change_point, maxima_after_change_point), key=lambda s: s[1]))
        sorted_years, sorted_maxima = list(sorted_years), np.array(sorted_maxima)
        year_top_score_max = self.year_from_top_score(sorted_years[-self.number_of_top_values:],
                                                      sorted_maxima[-self.number_of_top_values:], top_max=True)
        year_top_score_min = self.year_from_top_score(sorted_years[:self.number_of_top_values],
                                                      sorted_maxima[:self.number_of_top_values], top_max=False)
        score_difference = year_top_score_max - year_top_score_min
        return [score_difference, year_top_score_max, year_top_score_min]

    def year_from_top_score(self, top_sorted_years, top_sorted_maxima, top_max=None):
        raise NotImplementedError


class MeanScore(SortedScore):

    def year_from_top_score(self, top_sorted_years, top_sorted_maxima, top_max=None):
        return np.mean(top_sorted_years)


class MedianScore(SortedScore):

    def year_from_top_score(self, top_sorted_years, top_sorted_maxima, top_max=None):
        return np.median(top_sorted_years)


class WeigthedScore(SortedScore):

    def year_from_top_score(self, top_sorted_years, top_sorted_maxima, top_max=None):
        assert isinstance(top_max, bool)
        if not top_max:
            top_sorted_maxima = np.sum(top_sorted_maxima) - top_sorted_maxima
        weights = top_sorted_maxima / np.sum(top_sorted_maxima)
        return np.sum(weights * top_sorted_years)
