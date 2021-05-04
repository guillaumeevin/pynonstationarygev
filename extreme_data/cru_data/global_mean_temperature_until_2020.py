import os.path as op
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

from extreme_data.utils import DATA_PATH

txt_filename = 'HadCRUT5.0Analysis_gl.txt'
txt_filepath = op.join(DATA_PATH, "met_office_global_temp", txt_filename)


def _year_to_monthly_global_mean_temp_anomaly_wrt_1961_1990():
    """Global temperature dataformat
     for year = 1850 to endyear
  format(i5,13f7.3) year, 12 * monthly values, annual value
  format(i5,12i7) year, 12 * percentage coverage of hemisphere or globe
    :return:
    """
    year_to_monthly_values = OrderedDict()
    assert op.exists(txt_filepath)
    with open(txt_filepath, "r") as f:
        for i, l in enumerate(f):
            if i % 2 == 0:
                year, *temps = l.split()
                last_temp = float(temps[-1])
                year = int(year)
                if year <= 2020:
                    monthly_values = [float(e) for e in temps[:12]]
                    year_to_monthly_values[year] = np.array(monthly_values)
    return year_to_monthly_values


def _year_to_monthly_global_mean_temp_anomaly_wrt_1850_1900():
    year_to_monthly_values = _year_to_monthly_global_mean_temp_anomaly_wrt_1961_1990()
    # Compute the average temperature for the reference period
    mean_temp_reference = np.mean([np.mean(year_to_monthly_values[year]) for year in range(1850, 1901)])
    # Remove this mean temp from all the data
    year_to_monthly_values_shifted = OrderedDict()
    for year, monthly_values in year_to_monthly_values.items():
        year_to_monthly_values_shifted[year] = monthly_values - mean_temp_reference
    return year_to_monthly_values_shifted


def _year_to_average_global_mean_temp(raw_data=None, spline=True):
    if raw_data is None:
        d = winter_year_to_averaged_global_mean_temp_wrt_1850_1900(spline)
    elif raw_data:
        d = _year_to_monthly_global_mean_temp_anomaly_wrt_1961_1990()
    else:
        d = _year_to_monthly_global_mean_temp_anomaly_wrt_1850_1900()
    years, average = [], []
    for year, monthly_data in d.items():
        years.append(year)
        average.append(np.mean(monthly_data))
    return years, average


def winter_year_to_averaged_global_mean_temp_wrt_1850_1900(spline=True):
    years = []
    average_list = []
    year_to_monthly_data = _year_to_monthly_global_mean_temp_anomaly_wrt_1850_1900()
    for year in range(1851, 2021):
        # data from the past year
        monthly_data_past_year = year_to_monthly_data[year - 1][-5:]
        # data from the current year
        monthly_data_current_year = year_to_monthly_data[year][:7]
        all_monthly_data = np.concatenate([monthly_data_past_year, monthly_data_current_year])
        assert len(all_monthly_data) == 12
        average = np.mean(all_monthly_data)
        years.append(year)
        average_list.append(average)
    # Apply spline if needed
    if spline:
        years, average_list = np.array(years), np.array(average_list)
        f = UnivariateSpline(years, average_list, s=3, w=None)
        average_list = f(years)
    return OrderedDict(zip(years, average_list))


def main_plot_mean_temperature():
    ax = plt.gca()
    for spline in [True, False]:
        res = _year_to_average_global_mean_temp(raw_data=None, spline=spline)
        # res = winter_year_to_averaged_global_mean_temp_wrt_1850_1900()
        years, average = res
        plt.plot(years, average)
        ax.set_xlabel('Winter years')
        ax.set_ylabel('Anomaly of global mean temperature\nwith respect to pre-industrial levels (1850-1900)')
    plt.show()


if __name__ == '__main__':
    # print(_year_to_monthly_global_mean_temp_anomaly_wrt_1961_1990())
    main_plot_mean_temperature()
