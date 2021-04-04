from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy


class WeightComputationException(Exception):
    pass


class ReturnLevelComputationException(WeightComputationException):
    pass


class NllhComputationException(WeightComputationException):
    pass


class AbstractIndicator(object):

    @classmethod
    def get_indicator(cls, study: AbstractStudy, massif_name, bootstrap=False):
        raise NotImplementedError

    @classmethod
    def str_indicator(cls):
        raise NotImplementedError


class AnnualMaximaMeanIndicator(AbstractIndicator):

    @classmethod
    def get_indicator(cls, study: AbstractStudy, massif_name, bootstrap=False):
        if bootstrap:
            raise NotImplementedError
        else:
            return study.massif_name_to_annual_maxima[massif_name].mean()

    @classmethod
    def str_indicator(cls):
        return 'Mean annual maxima'


class ReturnLevel30YearsIndicator(AbstractIndicator):

    @classmethod
    def get_indicator(cls, study: AbstractStudy, massif_name, bootstrap=False):
        try:
            if bootstrap:
                return study.massif_name_to_return_level_list_from_bootstrap(return_period=30)[massif_name]
            else:
                return study.massif_name_to_return_level(return_period=30)[massif_name]
        except KeyError:
            raise ReturnLevelComputationException

    @classmethod
    def str_indicator(cls, bootstrap):
        return '30-year return level'