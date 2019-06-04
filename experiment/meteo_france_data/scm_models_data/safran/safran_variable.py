import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class SafranSnowfallVariable(AbstractVariable):
    """"
    Safran data is hourly

    Hypothesis:

    -How to count how much snowfall in one hour ?
        I take the average between the rhythm of snowfall per second between the start and the end
        and multiply that by 60 x 60 which corresponds to the number of seconds in one hour

    -How do how I define the limit of a day ?
        From the start, i.e. in August at 4am something like that,then if I add a 24H duration, I arrive to the next day

    -How do you aggregate several days ?
        We aggregate all the N consecutive days into a value x_i, then we take the max
        (but here the problem might be that the x_i are not idnependent, they are highly dependent one from another)
    """

    NAME = 'Snowfall'
    UNIT = 'kg per m2 or mm'

    @classmethod
    def keyword(cls, nb_consecutive_days=3):
        return 'Snowf'

    def __init__(self, variable_array, nb_consecutive_days):
        super().__init__(variable_array)
        self.nb_consecutive_days_of_snowfall = nb_consecutive_days
        # Compute the daily snowfall in kg/m2
        snowfall_rates = variable_array

        # Compute the mean snowrate, then multiply it by 60 * 60 * 24
        # day_duration_in_seconds = 24 * 60 * 60
        # nb_days = len(snowfall_rates) // 24
        # print(nb_days)
        # daily_snowrate = [np.mean(snowfall_rates[24 * i:24 * (i + 1) + 1], axis=0) for i in range(nb_days)]
        # self.daily_snowfall = day_duration_in_seconds * np.array(daily_snowrate)

        # Compute the hourly snowfall first, then aggregate
        mean_snowfall_rates = 0.5 * (snowfall_rates[:-1] + snowfall_rates[1:])
        hourly_snowfall = 60 * 60 * mean_snowfall_rates
        # Transform the snowfall amount into a dataframe
        nb_days = len(hourly_snowfall) // 24
        self.daily_snowfall = [sum(hourly_snowfall[24 * i:24 * (i + 1)]) for i in range(nb_days)]

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        # Aggregate the daily snowfall by the number of consecutive days
        shifted_list = [self.daily_snowfall[i:] for i in range(self.nb_consecutive_days_of_snowfall)]
        # First element of shifted_list is of length n, Second element of length n-1, Third element n-2....
        # The zip is done with respect to the shortest list
        snowfall_in_consecutive_days = np.array([sum(e) for e in zip(*shifted_list)])
        # The returned array is of size n-nb_days+1 x nb_massif
        # so that the length of the vector match a year, we can add zeros (since it corresponds to the July month,
        # we are sure that there is no snowfall at this period) However such trick does not work for other variable such as Temperature
        nb_days_in_a_year = len(self.daily_snowfall)
        nb_days_in_vector, nb_altitudes = snowfall_in_consecutive_days.shape
        zeros_to_add = np.zeros([nb_days_in_a_year - nb_days_in_vector, nb_altitudes])
        snowfall_in_consecutive_days = np.concatenate([snowfall_in_consecutive_days, zeros_to_add])
        return snowfall_in_consecutive_days


class SafranRainfallVariable(SafranSnowfallVariable):
    NAME = 'Rainfall'
    UNIT = 'kg per m2 or mm'

    @classmethod
    def keyword(cls, nb_consecutive_days=3):
        return 'Rainf'


class SafranTotalPrecipVariable(AbstractVariable):

    def __init__(self, snow_variable_array, rain_variable_array, nb_consecutive_days):
        super().__init__(None)
        self.snow_precipitation = SafranSnowfallVariable(snow_variable_array, nb_consecutive_days)
        self.rain_precipitation = SafranRainfallVariable(rain_variable_array, nb_consecutive_days)

    @classmethod
    def keyword(cls, nb_consecutive_days=3):
        return [SafranSnowfallVariable.keyword(), SafranRainfallVariable.keyword()]

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.snow_precipitation.daily_time_serie_array + self.rain_precipitation.daily_time_serie_array


class SafranTemperatureVariable(AbstractVariable):
    NAME = 'Temperature'
    UNIT = 'Celsius Degrees'

    @classmethod
    def keyword(cls, nb_consecutive_days=3):
        return 'Tair'

    def __init__(self, variable_array):
        super().__init__(variable_array)
        # Temperature are in K, I transform them as celsius
        self.hourly_temperature = self.variable_array - 273.15
        nb_days = len(self.hourly_temperature) // 24
        # Compute the mean temperature
        self.daily_temperature = [np.mean(self.hourly_temperature[24 * i:24 * (i + 1)], axis=0) for i in range(nb_days)]

    @property
    def daily_time_serie_array(self):
        return np.array(self.daily_temperature)