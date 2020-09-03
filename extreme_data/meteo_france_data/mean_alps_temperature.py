import pandas as pd
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranTemperature


def show_year_to_mean_alps_temperatures():
    ax = plt.gca()
    for altitude in range(900, 3300, 300):
    # for altitude in [900]:
        year_to_mean_alps_temperature = load_year_to_mean_alps_temperatures(altitude)
        ax.plot(year_to_mean_alps_temperature.keys(), year_to_mean_alps_temperature.values(), label=altitude)
    ax.legend()
    plt.show()


def load_year_to_mean_alps_temperatures(altitude=900):
    study = SafranTemperature(altitude=altitude)
    df = pd.DataFrame.from_dict(data=study.year_to_annual_mean).transpose().mean(axis=1)  # type: pd.Series
    df.index = df.index.astype(float)
    year_to_mean_alps_temperature = df.to_dict()
    return year_to_mean_alps_temperature


if __name__ == '__main__':
    show_year_to_mean_alps_temperatures()
