import numpy as np
import logging as log
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from windpowerlib import ModelChain, WindTurbine
from windpowerlib import data as wt

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
    df_forecast = df_forecast[['wind_speed', 'pressure', 'temp']]
    df_forecast['pressure'] = df_forecast['pressure'].mul(100) #convert to Pa
    df_forecast = df_forecast.rename(columns= {'temp': 'temperature'})

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
    :return: dataframe with all the information for power production modelling
    """

    log.debug("Recalculating forecasted wind speed at anemometer height to wind speed at rotor height")
    forecast_velocity['roughness_length'] = roughness_coefficient
    forecast_velocity['wind_speed_100'] = forecast_velocity['wind_speed']* pow((height_rotor/height_forecast)
                                                                                         ,roughness_coefficient)

    arrays = [['wind_speed', 'pressure', 'temperature', 'roughness_length', 'wind_speed_100'],[10,0,2,0,100]]
    forecast_velocity.columns = pd.MultiIndex.from_arrays(arrays, names=('variable', 'height'))
    forecast_velocity = forecast_velocity.rename(columns= {'wind_speed_100': 'wind_speed'})

    return forecast_velocity

def get_turbine_library():
    """
    Get all the turbines available in the windpowerlib database
    :return: dataframe with all the turbine names available in the database
    """
    turbine_database = wt.get_turbine_types(print_out=False)

    return turbine_database

def iterate_turbine_library(turbine_database: pd.DataFrame, weather_forecast: pd.DataFrame):
    """
    Function iterating over all the turbines available in the database, calculating power output for given forecast
    :param turbine_database: dataframe with all the turbines available in the database
    :param weather_forecast: dataframe with hourly weather forecast
    :return: None
    """
    results = []
    for index, row in turbine_database.iterrows():
        result_dict = {}
        turbine = initialize_wind_turbine(row['turbine_type'])
        results_turbine = calculate_power_output(turbine, weather_forecast)
        result_dict['turbine_type'] = row['turbine_type']
        result_dict['power_output_hourly'] = results_turbine.power_output
        result_dict['energy_produced_kWh'] = round(results_turbine.power_output.sum(),2)
        results.append(result_dict)

    max_idx = find_maximum_power(results)
    plot_power_production(results[max_idx], weather_forecast)

def initialize_wind_turbine(turbine_type):
    """
    Initialize parameters of turbine from database with given turbine type
    :param turbine_type: name of turbine
    :return: turbine parameters
    """
    turbine_args = {
        'turbine_type': turbine_type,  # turbine type as in oedb turbine library
        'hub_height': 120
    }
    turbine = WindTurbine(**turbine_args)

    return turbine

def calculate_power_output(turbine, weather_forecast):
    """
    Calculate predicted horly power output for selected wind turbine type
    :param turbine: turbine type from the database
    :param weather_forecast: predicted hourly weather conditions from the weather API
    :return: hourly power output from selected turbine, in KW
    """

    modelchain_data = {
        'wind_speed_model': 'logarithmic',      # 'logarithmic' (default),'hellman' or 'interpolation_extrapolation'
        'density_model': 'barometric',           # 'barometric' (default), 'ideal_gas' or 'interpolation_extrapolation'
        'temperature_model': 'linear_gradient', # 'linear_gradient' (def.) or 'interpolation_extrapolation'
        'power_output_model':
            'power_curve',          # 'power_curve' (default) or 'power_coefficient_curve'
        'density_correction': False,             # False (default) or True
        'obstacle_height': 0,                   # default: 0
        'hellman_exp': None}                    # None (default) or None

    model_turbine = ModelChain(turbine, **modelchain_data).run_model(weather_forecast)
    turbine.power_output = model_turbine.power_output/1000 #return hourly power output in kW

    #print(f"Calculated power output for {turbine.turbine_type} is {round(turbine.power_output.sum(),2)} kWh")

    return turbine

def find_maximum_power(results: list):
    """
    Function returning parameters of the turbine giving the maximum output in selected conditions
    :param results: list with all predicted power outputs for turbines in the database
    :return: None
    """
    results = [{k: v for k, v in d.items() if k != 'power_output_hourly'} for d in results]
    df_results = pd.DataFrame(results)

    max_power_idx = df_results['energy_produced_kWh'].idxmax()
    print(f"The turbine with the greatest power output is {df_results['turbine_type'].iloc[max_power_idx]} "
          f"and would produce {round(df_results['energy_produced_kWh'].iloc[max_power_idx]/1000,2)} MWh of electric energy"
          f" in the upcoming 4 days")

    return max_power_idx

def plot_power_production(wind_production: dict, forecast: pd.DataFrame):
    """
    Plot results of wind forecast and predicted power output
    :param wind_production: predicted hourly power output
    :param forecast: wind prediction
    :return: None
    """
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(2)

    hours = mdates.HourLocator(interval = 4)
    h_fmt = mdates.DateFormatter('%d/%m %H:%M')

    ax[0].plot(forecast[('wind_speed',100)].index, forecast[('wind_speed',100)].values, color = 'red')
    ax[0].set_title(f"Predicted weather conditions on the height of 100 m")
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Wind speed, m/s')
    ax[0].xaxis.set_major_locator(hours)
    ax[0].xaxis.set_major_formatter(h_fmt)

    ax[1].plot(wind_production['power_output_hourly'].index, wind_production['power_output_hourly'], color = 'green')
    ax[1].set_title(f"Predicted power production for turbine {wind_production['turbine_type']}")
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Power output, kW')
    ax[1].xaxis.set_major_locator(hours)
    ax[1].xaxis.set_major_formatter(h_fmt)
    fig.autofmt_xdate(rotation=45)
    plt.show()

if __name__ == "__main__":
    forecast_list  = get_forecast(54.335472056066855, 16.564964098411, "ccabb6fe85ca3bac888578fe83955ed2")
    forecast_48hrs = clean_data(forecast_list)
    forecast_48hrs = calculate_vertical_wind_distribution(roughness_coefficient= 0.22, height_rotor=100.,
                                                        height_forecast=10.,forecast_velocity=forecast_48hrs)
    turbine_database= get_turbine_library()
    iterate_turbine_library(turbine_database,forecast_48hrs)
