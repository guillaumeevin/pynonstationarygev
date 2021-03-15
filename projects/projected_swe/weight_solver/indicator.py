class AbstractIndicator(object):

    @classmethod
    def get_indicator(cls, study, bootstrap=False):
        raise NotImplementedError


class AnnualMaximaMeanIndicator(AbstractIndicator):

    @classmethod
    def get_indicator(cls, study, bootstrap=False):
        pass


class ReturnLevelIndicator(AbstractIndicator):

    @classmethod
    def get_indicator(cls, study, bootstrap=False):
        pass
