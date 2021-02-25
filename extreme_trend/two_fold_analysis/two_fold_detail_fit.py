from typing import List

import numpy as np

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_trend.two_fold_analysis.utils import Grouping, get_key_with_min_value, Score
from spatio_temporal_dataset.slicer.split import Split


class TwoFoldMassifFit(object):

    def __init__(self, model_classes, list_two_fold_datasets, **kargs):
        self.model_classes = model_classes
        self.sample_id_to_sample_fit = {
            sample_id: TwoFoldSampleFit(model_classes, two_fold_datasets=two_fold_datasets, **kargs)
            for sample_id, two_fold_datasets in enumerate(list_two_fold_datasets)
        }

    def best_model(self, score, group):
        if group is Grouping.MEAN_RANKING:
            return get_key_with_min_value(self.model_class_to_mean_ranking(score))
        else:
            raise NotImplementedError

    def sample_id_to_ordered_model(self, score):
        return {
            sample_id: sample_fit.ordered_model_classes(score)
            for sample_id, sample_fit in self.sample_id_to_sample_fit.items()
        }

    def model_class_to_scores(self):
        pass

    def model_class_to_rankings(self, score):
        model_class_to_ranks = {model_class: [] for model_class in self.model_classes}
        for ordered_model in self.sample_id_to_ordered_model(score=score).values():
            for rank, model_class in enumerate(ordered_model):
                model_class_to_ranks[model_class].append(rank)
        return model_class_to_ranks

    def model_class_to_mean_ranking(self, score):
        return {
            model_class: np.mean(ranks)
            for model_class, ranks in self.model_class_to_rankings(score).items()
        }


class TwoFoldSampleFit(object):

    def __init__(self, model_classes, **kargs):
        self.model_classes = model_classes
        self.model_class_to_model_fit = {
            model_class: TwoFoldModelFit(model_class, **kargs) for model_class in self.model_classes
        }

    def ordered_model_classes(self, score):
        # Always ordered from the lower score to the higher score.
        key = lambda model_class: self.model_class_to_model_fit[model_class].score(score)
        return sorted(self.model_classes, key=key)

    def scores(self, score):
        return [self.model_class_to_model_fit[model_class].score(score) for model_class in self.model_classes]


class TwoFoldModelFit(object):

    def __init__(self, model_class, two_fold_datasets, fit_method):
        self.model_class = model_class
        self.fit_method = fit_method
        self.estimators = [fitted_linear_margin_estimator_short(model_class=self.model_class, dataset=dataset,
                                                                fit_method=self.fit_method)
                           for dataset in two_fold_datasets]  # type: List[LinearMarginEstimator]
        self.estimator_fold_1 = self.estimators[0]
        self.estimator_fold_2 = self.estimators[1]

    def score(self, score):
        if score == Score.NLLH_TEST:
            return self.nllh_test_temporal
        else:
            raise NotImplementedError

    @property
    def nllh_test_temporal(self):
        return sum([e.nllh(split=Split.test_temporal) for e in self.estimators])
