import time


class AbstractEstimator(object):

    def __init__(self):
        self.fit_duration = None

    def timed_fit(self):
        ts = time.time()
        result = self.fit()
        te = time.time()
        log_time = int((te - ts) * 1000)
        self.fit_duration = log_time
        return result

    def fit(self):
        pass

    def error(self, true_max_stable_params: dict):
        pass