

class AbstractVariable(object):

    def __init__(self, dataset):
        self.dataset = dataset

    @property
    def daily_time_serie(self):
        # Return an array of size length of time series x nb_massif
        raise NotImplementedError