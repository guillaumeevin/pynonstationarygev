
# Eurocode quantile str
EUROCODE_RETURN_LEVEL_STR = '50-year return level of GSL (kN $m^-2$)'
# Eurocode quantile correspond to a 50 year return period
EUROCODE_QUANTILE = 0.98
# Altitudes (between low and mid altitudes) < 2000m and should be > 200m
EUROCODE_ALTITUDES = [300, 600, 900, 1200, 1500, 1800]
#  Last year taken into account for the Eurocode
# Date of publication was 2014, therefore the winter 2013/2014 could not have been measured
# Therefore, the winter 2012/2013 was the last one. Thus, 2012 is the last year for the Eurocode
LAST_YEAR_FOR_EUROCODE = 2012
# Year of interest for the EUROCODE
YEAR_OF_INTEREST_FOR_RETURN_LEVEL = 2017