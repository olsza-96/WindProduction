"""Script predicting energy production from wind energy given specific wind turbine with 5 day forecast
List of used important variables:
weather_url- input of the weather url with given ID
forecast_url- the same for forecast
velocity - array containing velocities of the wind from the forecast
orientation - array containing wind orientation from the forecast
weat_forecast - array containing forecast for the upcoming 5 days: date, hour, wind speed, wind orientation
power_production_curve - array containing power curve for given turbine
power_speed - array containing power corresponding to wind speeds from the forecast
hours - array containing times of blowing with given speeds from the forecast
energy_out - array containing energy production for each speed coming from the forecast 40 x 41
energy_per_3hours - array containing energy production for each 3hour step from the forecast 40 x 1
energy_all - the overall energy output for the upcoming 5 days
scale - contains scale factor for each mean wind speed used to calculate rayleigh distribution
rayleigh1 - array containing rayleigh probability distribution for given scale, 40 columns/rows
wind_velocity_real - array containing real wind velocity taking into consideration height of the turbine
"""

import numpy as np
import json, requests
from datetime import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.ticker import FormatStrFormatter

 
def get_weather():
    # Fetch weather data (example)
    # Warszawa http://api.openweathermap.org/data/2.5/weather?q=Warszawa,pl&appid=7c739b6b8c2d0b34ea2eddce5de98874
    weather_url = "http://api.openweathermap.org/data/2.5/weather?q=Warszawa,pl&appid=7c739b6b8c2d0b34ea2eddce5de98874"
    weather_json = requests.get(weather_url)
    forecast_url = "http://api.openweathermap.org/data/2.5/forecast?q=Warszawa,pl&appid=7c739b6b8c2d0b34ea2eddce5de98874"
    forecast_json = requests.get(forecast_url)
    
    # Convert JSON data to a Python dictionary
    weat = json.loads(weather_json.text)
    fore = json.loads(forecast_json.text)
    
    #print the weather forecast for 5 days, create matrix with the data
    temp1= []
    temp2= []
    temp3= []
    temp4=[]
    weat_fore_range = int(40)
    
    #create matrix with forecast data
    #format of the data columns: City I Date I Hour I Wind speed I Wind direction
    for i in range(weat_fore_range):
        ts = int(fore["list"][i]["dt"])
        temp1.append(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d')) #date
        temp2.append(datetime.utcfromtimestamp(ts).strftime('%H:%M:%S')) #time
        temp3.append(float((fore["list"][i]["wind"]["speed"]))) #wind speed
        temp4.append(float(fore["list"][i]["wind"]["deg"])) #wind orientation
    #changes to float type so that values can be used for calculation
    velocity=np.asarray(temp3)
    orientation=np.asarray(temp4)
    velocity.astype(float)
    orientation.astype(float)
    weat_fore=np.stack((temp1,temp2,temp3,temp4), axis=-1)

   #Append to CSV file (; instead of , for polish version of Excel)
   #format of the data columns: City I Date I Hour I Wind speed I Wind direction

    file=open("forecast_log.csv", "a+")
    for i in range(weat_fore_range):
        file.write(str(fore["city"]["name"]) + ";")
        ts = int(fore["list"][i]["dt"])
        file.write(datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d') + ";")
        file.write(datetime.utcfromtimestamp(ts).strftime('%H:%M:%S') + ";")
        file.write(str(fore["list"][i]["wind"]["speed"]) + ";")
        file.write(str(fore["list"][i]["wind"]["deg"]) + "\n")
        file.close

    return(weat_fore, velocity, orientation)
   
def wind_power(wind_vel, distribution, speeds, time): #calculates value of power for given wind velocity
    power_production_curve=[]
    #generates general power curve given by the manufacturer
    
    for i in wind_vel:
        if i<9 and i>=0:
            power_production_curve.append(0.2277*(i**2)-0.8124*i+0.4256)
            for j, x in enumerate(range(len(power_production_curve))):
                if power_production_curve[j]<0:
                    power_production_curve[j]=0
        elif i>=9 and i<=20:
            power_production_curve.append(12)
        elif i>20:
            power_production_curve.append(0)
#calculate power production with given rayleigh distribution
    power_speed=[]
    hours=[]
    for i,x in enumerate(distribution):
        hours.append(distribution[i]*time[i])

    for i in speeds:
        if i<9 and i>=0:
            power_speed.append(0.0166*(i**2)-0.0489*i+0.0083)
            for j, x in enumerate(range(len(power_speed))):
                if power_speed[j]<0:
                    power_speed[j]=0
        elif i>=9 and i<=20:
            power_speed.append(12)
        elif i>20:
            power_speed.append(0)

#array containing the energy output for each velocity
    energy_out=[]
    for i, x in enumerate(hours):
            energy_out.append(power_speed[j]*hours[i])

#array containing energy for each 3hours
    energy_per_3hours=np.sum(energy_out, axis=1)

#array containing all the energy forecasted
    energy_all=np.sum(energy_out)
    print ("The total energy production in the upcoming five days will be equal to "+ str(round(energy_all,2))+ " kWh")
    return(power_production_curve, power_speed, energy_out, energy_per_3hours, energy_all)

def rayleigh(wind_velocity): #create weibull distribution of powers for given mean velocity
    const=math.sqrt(2/math.pi)
    #scale zawiera wartosci do liczenia prawdopodobienstwa dla kazdej sredniej predkosci wiatru z prognozy
    scale=[]
    for i in range(len(wind_velocity)):
        scale.append(wind_velocity[i]*const)
    
#array containing rayleigh probability distribution for given scale
    rayleigh1=[]
    #speeds for which the probability will be calculated
    speeds=np.arange(0,20.5, 0.5)
    for j in range(len(scale)):
        for i in range(len(speeds)):
            const1=i/pow(scale[j],2)
            const2=-pow(i,2)
            const3=const2/(2*(pow(scale[j],2)))
            const4=math.exp(const3)
            const5=const1*const4
            rayleigh1.append(const5)
#reshape the probability distribution function to have matrix with 40 rows and 41 columns
#each row contains probabilty distribution for every 3hours
    rayleigh=np.reshape(rayleigh1, (40,41))
#calculates time of wind blowing with given velocity based on probability
    time=[]
    for i,x in enumerate(rayleigh):
        time.append(rayleigh[i]*3)
    time=np.asarray(time)

    return (rayleigh, speeds, time)

#rozklad pionowy predkosci wiatru
def wind_vertical_distribution(area_roughness, height_rotor, height_forecast, velocity):
    wind_velocity_real=[]
    constant= float((height_rotor/height_forecast))
    
    #loop calculating the real velocity at turbine's height
    for i, x in enumerate(range(len(velocity))):
        wind_velocity_real.append(velocity[i]*constant)

    return (wind_velocity_real)


def postprocessing (rayleigh, wind_velocity_real):
#draw probability distribution function for given hour graphs
    yaxis=rayleigh[:1]
    xaxis=np.arange(0, 20.5, 0.5)
    for i, x in enumerate(yaxis):
        yaxis=np.round(x,2)
    print(np.shape(yaxis))
    print(np.shape(xaxis))
    plt.scatter(xaxis, yaxis)
    plt.xlabel("Wind velocity, m/s")
    plt.ylabel("Rayleigh probability distribution")
    plt.title("Rayleigh probability distribution for mean velocity " +str(wind_velocity_real[:1])+ " m/s")
    plt.xlim(0,20)
    plt.ylim(0)
    plt.show()

def wind_orientation(orientation, wind, energy_3hour_fore):
    #calculating and printing wind orientations and energy production for all the directions
    north =[]
    east = []
    south = []
    west = []
    north_energy=[]
    east_energy=[]
    south_energy=[]
    west_energy=[]
    for i,x in enumerate(orientation):
        if x>=0 and x<=45 or x>=315 and x<=360:
            north.append(x)
            north_energy.append(energy_3hour_fore[i])
        elif x>45 and x<=135:
            east.append(x)
            east_energy.append(energy_3hour_fore[i])
        elif x>135 and x<=225:
            south.append(x)
            south_energy.append(energy_3hour_fore[i])
        elif x>225 and x<315:
            west.append(x)
            west_energy.append(energy_3hour_fore[i])

    print("Northern wind orientations are: " + str(north) + " degrees")
    print("Energy production from northern winds is equal to "+str(round(np.sum(north_energy),2))+ " kWh")
    print("Southern wind orientations are: " + str(south)+ " degrees")
    print("Energy production from southern winds is equal to "+str(round(np.sum(south_energy),2))+ " kWh")
    print("Western wind orientations are: " + str(west)+ " degrees")
    print("Energy production from western winds is equal to "+str(round(np.sum(west_energy),2))+ " kWh")
    print("Eastern wind orientations are: " + str(east)+ " degrees")
    print("Energy production from eastern winds is equal to "+str(round(np.sum(east_energy),2))+ " kWh")

def read_excel():
    #read historical data of power output from excel file
    with open("wind_dir.csv") as fx:
        historical_data = np.array([list(map(float, line.split(";"))) for line in fx.readlines()[1:]])
        historical_new=[]
        
        power=historical_data[:,1]
        power_new=[]
        speed_new=[]
        orientation_new=[]
        #leave the data that gives 0 power output
        for i in range(len(historical_data)):
            if power[i]!=0:
                power_new.append(historical_data[i,1])
                speed_new.append(historical_data[i,0])
                orientation_new.append(historical_data[i,2])
        historical_new=np.stack((speed_new,power_new,orientation_new), axis=-1) #create historical data not containing 0 power outputs
        
        north=[]
        south=[]
        east=[]
        west=[]
    #dividie production regarding wind orientation
        for i in range(len(historical_new)):
            if (historical_new[i,2])<=45 and (historical_new[i,2])>=315:
                north.append(historical_new[i])
            elif (historical_new[i,2])<=135 and (historical_new[i,2])>45:
                east.append(historical_new[i])
            elif (historical_new[i,2])<=225 and (historical_new[i,2])>135:
                south.append(historical_new[i])
            elif (historical_new[i,2])<315 and (historical_new[i,2])>225:
                west.append(historical_new[i])

        north=np.asarray(north)
        south=np.asarray(south)
        east=np.asarray(east)
        west=np.asarray(west)
        print(north)

    return(north,south,east,west)

def historical_power(north,south,east,west, power_curve,wind_vel):
    #printing graphs of power output from historical data 
    yaxis_n=north[:1] #different notation because north seems to be empty
    xaxis_n=north[:0]
    yaxis_e=east[:,1]
    xaxis_e=east[:,0]
    yaxis_s=south[:,1]
    xaxis_s=south[:,0]
    yaxis_w=west[:,1]
    xaxis_w=west[:,0]
    
    #draw theoretical power curve given by the manufacturer:
    velocity_range=np.arange(0,21,1)
    power_production_curve=[0,0,0.05,0.35, 0.93, 1.9, 3.47, 5.61, 8.27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    #plots of historical powers and theoretical power curve
    plt.scatter(xaxis_n, yaxis_n, label='North', color='y', marker='.')
    plt.scatter(xaxis_e, yaxis_e, label='East', color='r', marker='o')
    plt.scatter(xaxis_s, yaxis_s, label='South', color='b', marker='^')
    plt.scatter(xaxis_w, yaxis_w, label='West', color='g', marker='+')
    plt.plot(velocity_range,power_production_curve, label='Theoretical', color='k')
    plt.xlabel("Wind velocity, m/s")
    plt.ylabel("Power, kW")
    plt.title("Power output from turbine regarding wind orientation")
    plt.legend()
    plt.xlim(0,20)
    plt.ylim(0,15)
    plt.show()


if __name__ == "__main__":
    forecast_all, wind_vel_fore, wind_orient_fore =get_weather() #gets all the data for weather forecast
    roughness_coefficient=0.22 #from the paper, rather smooth area
    rotor_height = 15.
    forecast_height = 30.
    wind_velocity=[1,2,3,4,21,5,10] #matrix containing wind velocities from the weather forecast
    wind_velocity_real_distribution=wind_vertical_distribution(roughness_coefficient, rotor_height, forecast_height,wind_vel_fore)
    distribution, speeds, time =rayleigh(wind_velocity_real_distribution)
    power_curve, production, energy_per_velocity, energy_3hour_fore, energy_total =wind_power(wind_velocity_real_distribution, distribution, speeds, time)
    graphs= postprocessing(distribution, wind_velocity_real_distribution)
    wind_orient=wind_orientation(wind_orient_fore, wind_velocity_real_distribution, energy_3hour_fore)
    north_historical,south_historical,east_historical,west_historical=read_excel()
    power_graphs=historical_power(north_historical,south_historical,east_historical,west_historical, power_curve, wind_velocity_real_distribution)
