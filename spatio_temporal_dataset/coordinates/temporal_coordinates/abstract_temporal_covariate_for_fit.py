class AbstractTemporalCovariateForFit(object):

    @classmethod
    def get_temporal_covariate(cls, t):
        raise NotImplementedError


class TimeTemporalCovariate(AbstractTemporalCovariateForFit):

    @classmethod
    def get_temporal_covariate(cls, t):
        return t


class MeanGlobalTemperatureCovariate(AbstractTemporalCovariateForFit):

    @classmethod
    def get_temporal_covariate(cls, t):
        pass
