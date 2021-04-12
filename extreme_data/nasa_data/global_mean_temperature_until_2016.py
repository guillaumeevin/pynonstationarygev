"""
Source:
https://www.jpl.nasa.gov/edu/teach/activity/graphing-global-temperature-trends/


We took the csv file correspond to "Global annual mean temperature data"
"""
import pandas as pd

from root_utils import get_full_path

relative_path = r'local/NASA_data/global_annual_mean_temp_anomalies_land-ocean_1880-2016_modified.csv'
edf_filepath = get_full_path(relative_path=relative_path)

def load_year_to_mean_global_temperature_until_2016():
    df = pd.read_csv(edf_filepath)
    df = df.astype({'Year': 'float'})
    d = dict(zip(df['Year'], df['Actual Temp']))
    # # Cheat for the last years
    # print('Warning: using fake values for the last few years !!!')
    # for idx in range(1, 5):
    #     d[2016 + idx] = d[2016]
    return d


