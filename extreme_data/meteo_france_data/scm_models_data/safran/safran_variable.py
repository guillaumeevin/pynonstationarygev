import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class SafranSnowfallVariable(AbstractVariable):
    """"
    Safran data is hourly

    Hypothesis:

    -How to count how much snowfall in one hour ?
        I take the average between the rhythm of snowfall per second between the start and the end
        and multiply that by 60 x 60 which corresponds to the number of seconds in one hour

    -How do how I define the limit of a day ?
        From the start, i.e. in August at 6am,then if I add a 24H duration, I arrive to the next day

    -How do you aggregate several days ?
        We aggregate all the N consecutive days into a value x_i, then we take the max
        (but here the problem might be that the x_i are not idnependent, they are highly dependent one from another)
    """

    NAME = 'Snowfall'
    UNIT = 'kg m$^{-2}$'

    # this could have been mm w.e (mm in water equivalent)

    @classmethod
    def keyword(cls):
        return 'Snowf'

    def __init__(self, variable_array, nb_consecutive_days=3):
        super().__init__(variable_array)
        self.nb_consecutive_days_of_snowfall = nb_consecutive_days
        # Compute the daily snowfall in kg/m2
        mean_snowfall_rates = self.get_snowfall_rates(variable_array)
        hourly_snowfall = 60 * 60 * mean_snowfall_rates
        # Transform the snowfall amount into a dataframe
        nb_days = len(hourly_snowfall) // 24
        self.daily_snowfall = self.daily_snowfall(hourly_snowfall, nb_days)

    def get_snowfall_rates(self, variable_array):
        return 0.5 * (variable_array[:-1] + variable_array[1:])

    def daily_snowfall(self, hourly_snowfall, nb_days):
        return [sum(hourly_snowfall[24 * i:24 * (i + 1)]) for i in range(nb_days)]

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


class SafranSnowfallVariableNotCenterOnDay(SafranSnowfallVariable):
    NAME = 'Snowfall MeteoFranceRate 6hto5h'

    def get_snowfall_rates(self, variable_array):
        return variable_array[:-1]


class SafranSnowfallVariableCenterOnDay(SafranSnowfallVariable):
    NAME = 'Snowfall MeteoFranceRate CenterOnDay'

    def daily_snowfall(self, hourly_snowfall, nb_days):
        hourly_snowfall_without_first_and_last_days = hourly_snowfall[18:-6]
        assert len(hourly_snowfall_without_first_and_last_days) % 24 == 0
        daily_snowfall = super().daily_snowfall(hourly_snowfall[18:-5], nb_days - 2)
        zero_array = daily_snowfall[0] * 0
        daily_snowfall = [zero_array] + daily_snowfall + [zero_array]
        return daily_snowfall

    def get_snowfall_rates(self, variable_array):
        return variable_array[:-1]


class SafranSnowfallVariableCenterOnDayMeanRate(SafranSnowfallVariableCenterOnDay):
    NAME = 'Snowfall MyRate CenterOnDay'

    def get_snowfall_rates(self, variable_array):
        return 0.5 * (variable_array[:-1] + variable_array[1:])


class SafranDateFirstSnowfallVariable(SafranSnowfallVariable):
    NAME = 'Date First Snow'
    UNIT = 'days'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        daily_time_series = super().daily_time_serie_array
        new_daily_time_series = []
        for i, s in enumerate(daily_time_series.transpose()):
            dates_with_snow = np.nonzero(s)[0]
            if len(dates_with_snow) > 0:
                min = np.min(dates_with_snow)
            else:
                min = np.nan
            # first_date = 1 - min / 366
            first_date = 1 - min / 366
            first_date_repeated = np.ones(len(s)) * first_date
            new_daily_time_series.append(first_date_repeated)
        new_daily_time_series_array = np.array(new_daily_time_series).transpose()
        return new_daily_time_series_array


class SafranRainfallVariable(SafranSnowfallVariable):
    """Warning: this corresponds to water falling. Total precipitaiton equals Rainfall + Snowfall"""
    NAME = 'Rainfall'
    UNIT = 'kg m$^{-2}$'

    @classmethod
    def keyword(cls):
        return 'Rainf'


class SafranTotalPrecipVariable(AbstractVariable):
    NAME = 'Precipitation'[:6]
    UNIT = 'kg m$^{-2}$'

    def __init__(self, snow_variable_array, rain_variable_array, nb_consecutive_days=3):
        super().__init__(None)
        snow_precipitation = SafranSnowfallVariable(snow_variable_array, nb_consecutive_days)
        rain_precipitation = SafranRainfallVariable(rain_variable_array, nb_consecutive_days)
        self._daily_time_serie_array = snow_precipitation.daily_time_serie_array \
                                       + rain_precipitation.daily_time_serie_array

    @classmethod
    def keyword(cls):
        return [SafranSnowfallVariable.keyword(), SafranRainfallVariable.keyword()]

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self._daily_time_serie_array


class SafranNormalizedPrecipitationRateVariable(AbstractVariable):
    NAME = 'Normalized Precip'

    def __init__(self, temperature_variable_array, snow_variable_array, rain_variable_array, nb_consecutive_days=3):
        super().__init__(None)
        temperature = SafranTemperatureVariable(temperature_variable_array)
        total_precipitation = SafranTotalPrecipVariable(snow_variable_array, rain_variable_array, nb_consecutive_days)
        beta = 0.06
        self._daily_time_serie_array = np.exp(-beta * temperature.daily_time_serie_array) \
                                       * total_precipitation.daily_time_serie_array

    @classmethod
    def keyword(cls):
        return [SafranTemperatureVariable.keyword(), SafranSnowfallVariable.keyword(), SafranRainfallVariable.keyword()]

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self._daily_time_serie_array


class SafranNormalizedPrecipitationRateOnWetDaysVariable(SafranNormalizedPrecipitationRateVariable):

    def __init__(self, temperature_variable_array, snow_variable_array, rain_variable_array, nb_consecutive_days=3):
        super().__init__(temperature_variable_array, snow_variable_array, rain_variable_array, nb_consecutive_days)
        total_precipitation = SafranTotalPrecipVariable(snow_variable_array, rain_variable_array, nb_consecutive_days)
        mask_for_nan_values = total_precipitation.daily_time_serie_array < 0.01
        self._daily_time_serie_array[mask_for_nan_values] = np.nan


class SafranTemperatureVariable(AbstractVariable):
    NAME = 'Temperature'
    UNIT = '$^oC$'

    @classmethod
    def keyword(cls):
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
