import numpy as np
import logging as log
import requests
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as p
import csv
import seaborn as sns
from matplotlib.dates import DateFormatter

log.getLogger().setLevel(log.INFO)
log.basicConfig(format="%(asctime)s - [%(levelname)s]: %(message)s", datefmt="%H:%M:%S")

def get_forecast(lat: float, lon:float, API_key:str):
    """
    Get hourly forecast for 48 hrs from Openweathermap API for chosen location

    :param lat: latitude of place for which getting forecast, degrees
    :param lon: longitude of place for which getting forecast, degrees
    :param API_key: user key for the account at https://openweathermap.org
    :return: list of hourly forecast for the following 48 hrs
    """

    log.debug(f"Getting hourly 48 hrs forecast for the coordinates lat: {round(lat,2)} lon: {round(lon,2)}")

    response: requests.Response = requests.get(f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,daily,alerts&appid={API_key}")
    log.debug(f"The response from the server: {response.status_code}")
    forecast =  response.json()['hourly']

    return forecast


def clean_data(forecast: list):
    """
    Converts input data to df and gets only relevant data for predicting wind production
    :param forecast: list of predictions for each hour in the following 48 hrs
    :return: clean dataframe
    """
    df_forecast = pd.DataFrame(forecast)
    df_forecast['dt'] = pd.to_datetime(df_forecast['dt'], unit='s')
    df_forecast = df_forecast.set_index(['dt'], drop=True)
    df_forecast = df_forecast.drop(columns= [x for x in df_forecast.columns if x not in ['wind_speed', 'wind_deg']])
    df_forecast = df_forecast.rename(columns= {'wind_speed': 'Forecast_wind_speed'})
    return df_forecast


def calculate_vertical_wind_distribution(roughness_coefficient:float, height_rotor: float,
                                         height_forecast:float, forecast_velocity:pd.DataFrame):

    """
    Returns predicted velocities on height of rotor
    http://green-power.com.pl/pl/home/wiatr-i-jego-pomiar-w-energetyce-wiatrowej/
    :param area_roughness: roughness coefficient for chosen location
    :param height_rotor: height of rotor, m
    :param height_forecast: height of anemometer, m
    :param forecast_velocity: dataframe with predicted parameters
    :return: dataframe with additional column containing wind velocities at rotor height
    """

    log.debug("Recalculating forecasted wind speed at anemometer height to wind speed at rotor height")

    forecast_velocity['Real_wind_speed'] = forecast_velocity['Forecast_wind_speed']* pow((height_rotor/height_forecast)
                                                                                         ,roughness_coefficient)

    return forecast_velocity


def get_power_curve(file_name: str):
    """
    Read power curve values for chosen type of turbine
    from https://www.thewindpower.net/turbine_en_33_vestas_v90-3000.php
    here used Vestas V112/3000
    :param file_name: file with power curve for chosen turbine
    :return: dict with power curve
    """
    
    power_curve_path: p.Path = p.Path.cwd().joinpath(file_name)

    with power_curve_path.open(mode="r", encoding="utf-8") as read_file:
        file_reader = csv.DictReader(read_file)

        dict_ = list(file_reader)[0]
        power_curve = dict(dict_)

    for k,v in power_curve.items():
        power_curve[k] = float(v)

    return power_curve

def calculate_wind_production(power_curve: dict, forecast: pd.DataFrame):

    """
    Calculate wind production for given turbine and hourly forecast
    :param power_curve: dict with power for each wind speed
    :param forecast: input df with forecast data
    :return: power production in kWh
    """

    log.debug("Calculating forecasted hourly energy production for the following 48 hrs")
    power_production = []
    for wind_speed in forecast['Real_wind_speed']:
        power_production.append(power_curve[str(round(wind_speed*2)/2)])

    forecast['Power_production'] = power_production

    return forecast


def wind_orientation(forecast: pd.DataFrame):
    """
    Adding info on wind direction for each wind speed 
    :param forecast: input df with forecast data
    :return: info on direction of wind blowing
    """
    #calculating and printing wind orientations and energy production for all the directions
    northern_wind = forecast.query("(wind_deg>=0 & wind_deg<=45) | \
                                   (wind_deg >= 315 & wind_deg <=360)").index
    eastern_wind = forecast.query("wind_deg>45 & wind_deg<=135").index
    southern_wind = forecast.query("wind_deg>135 & wind_deg<=225").index
    western_wind = forecast.query("wind_deg>225 & wind_deg<315").index

    forecast.loc[northern_wind, 'wind_orientation'] = "northern"
    forecast.loc[eastern_wind, 'wind_orientation'] = "eastern"
    forecast.loc[southern_wind, 'wind_orientation'] = "southern"
    forecast.loc[western_wind, 'wind_orientation'] = "western"

    return forecast

def statistics(forecast: pd.DataFrame, cut_in_speed: float, cut_off_speed: float):

    """
    Print statistics of predicted power production
    :param forecast: df with info on forecasted production
    :param cut_in_speed: minimum wind speed at which turbine is able to generate electricity
    :param cut_off_speed: maximum wind speed at which turbine can work
    """
    no_energy_produced = len(forecast.query(f"Real_wind_speed < {cut_in_speed}  | \
                                   Real_wind_speed > {cut_off_speed} ").index)

    log.info(f"Forecast was computed for the following dates: {forecast.index.min()} - {forecast.index.max()}")
    log.info(f"The forecasted power  = {round(forecast['Power_production'].sum()/1000,2)} MWh")
    log.info(f"Wind parameters for the following dates are min = {round(forecast['Real_wind_speed'].min(),2)} m/s"
             f"max = {round(forecast['Real_wind_speed'].max(),2)} m/s"
             f"mean = {round(forecast['Real_wind_speed'].mean(),2)} m/s")
    log.info(f"Number of hours with no energy produced {no_energy_produced}")

def plotting(forecast: pd.DataFrame):
    """
    Plot hourly energy production for the following 48 hrs
    and wind speed histogram by wind direction
    :param forecast: df with info on forecast
    """
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(2)

    df_agg = forecast.loc[:, ['Real_wind_speed', 'wind_orientation']].groupby('wind_orientation')
    vals = [df['Real_wind_speed'].values.tolist() for i, df in df_agg]

    # Draw
    colors = [plt.cm.Set1(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = ax[0].hist(vals, 60, stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    ax[0].legend({group:col for group, col in zip(np.unique(forecast['wind_orientation']).tolist(), colors[:len(vals)])})
    ax[0].set_title(f"Histogram of wind speed by wind direction")
    ax[0].set_ylabel("Number of hours ")
    ax[0].set_xticks(ticks=bins[::3])
    ax[0].set_xticklabels([round(b,1) for b in bins[::3]])

    ax[1].plot(forecast.index, forecast['Power_production']/1000, c='lightcoral')
    ax[1].set_title(f"Forecasted hourly energy production")
    ax[1].set_xlabel("Date")
    ax[1].set_ylabel("Energy production, MWh")
    ax[1].xaxis.set_major_formatter(DateFormatter('%d %b %H'))
    plt.setp(ax[1].get_xticklabels(), rotation=30, horizontalalignment='right')


    plt.show()

if __name__ == "__main__":
    forecast_list  = get_forecast(54.335472056066855, 16.564964098411, "ccabb6fe85ca3bac888578fe83955ed2")
    forecast_48hrs = clean_data(forecast_list)
    forecast_48hrs = calculate_vertical_wind_distribution(roughness_coefficient= 0.22, height_rotor=150.,
                                                        height_forecast=10.,forecast_velocity=forecast_48hrs)
    power_curve = get_power_curve('power_curve.csv')
    forecast_48hrs = calculate_wind_production(power_curve, forecast_48hrs)
    forecast_48hrs = wind_orientation(forecast_48hrs)
    statistics(forecast_48hrs, cut_in_speed=3.5, cut_off_speed=25.)
    plotting(forecast_48hrs)