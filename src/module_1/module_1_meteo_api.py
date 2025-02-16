import matplotlib.pyplot as plt
import requests #llamada a API
import time
import logging 
import json
import pandas as pd
from urllib.parse import urlencode
from typing import Any, Dict, Optional, List

#logger en vez de print

logger = logging.getLogger(__name__)

logger.level = logging.INFO


API_URL = "https://archive-api.open-meteo.com/v1/archive?"

COORDINATES = {
 "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
 "London": {"latitude": 51.507351, "longitude": -0.127758},
 "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum",
              "wind_speed_10m_max"]

#api

headers = {}
retries = 10 #API, intentos de hacer la llamada
cooldown = 2 #API, espera antes de volver

def call_api(
    API_URL: str,
    headers: Dict[str, any],
    retries: int = retries,
    ):
    
    for attempt in range(retries):
        try: 
            response = requests.get(API_URL, headers = headers)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.ConnectionError as e:
            logging.warning(f"Error en la llamada a la API (intento {attempt + 1} de {retries}): {e}")

        except requests.exceptions.HTTPError as er:
                logging.error(f"Error HPPT: {er}")
        
        time.sleep(cooldown)
    
    logging.error("No hay respuesta exitosa")
    return None
#no meto params porque esos van en la url y cuando uso la función
#ya los meto en ella directamente.    

def get_data_meteo_api(
    longitude: float,
    latitude: int,
    start_date: str, 
    end_date: str
    ):
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(VARIABLES),
        "timezone": "Europe/Madrid"
    }

    return call_api(API_URL + urlencode(params, safe=","), headers)

#plot

def modify_weather_data(data, variables: List[str]) -> pd.DataFrame:
    
    results = []
    data["time"] = pd.to_datetime(data["time"])
    
    grouped = data.groupby([data["city"], data["time"].dt.to_period("M")])
    '''
    for city, city_data in data.items():

        df = pd.DataFrame(city_data)
        #df["time"] = pd.to_datetime(df["time"])
        df.set_index('time', inplace=True)
        new_df = df.resample("M").mean()
        results[city] = new_df
    '''
    for (city, month), group in grouped:
        monthly_stats = {"city":city, "month": month.to_timestamp()}
        
        for variable in variables:
            monthly_stats[f"{variable}_mean"] = group[variable].mean()

        results.append(monthly_stats)

    return pd.DataFrame(results)

def graphic(data: pd.DataFrame):

    rows = len(VARIABLES)
    cols= len(data["city"].unique())

    fig, axes =plt.subplots(rows, cols, figsize=( 5*rows, 5*cols))
    
    for i, variable in enumerate(VARIABLES):
        for j, city in enumerate(data["city"].unique()):
            city_data = data[data["city"] == city]
            ax = axes[i,j]
            ax.plot(
                city_data["month"], 
                city_data[f"{variable}_mean"],
                label=f"{city} (mean)",
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("Value")
            ax.set_title(f"{city}-{variable}")
            ax.legend()
            ax.grid()


    plt.tight_layout()
    plt.savefig(
        "climate_charts.png", bbox_inches='tight'
    )


#main
def main():
    all_data = []
    start_date = "2010-01-01"
    end_date = "2020-12-31"
    time_spam = (
        pd.date_range(start_date, end_date, freq ="D").strftime("%Y-%m-%d").tolist()
    )

    for city, coordinates in COORDINATES.items():
        latitude = coordinates["latitude"]
        longitude = coordinates["longitude"]
        data = pd.DataFrame(
            get_data_meteo_api(longitude, latitude, start_date, end_date)["daily"]    
        ).assign(city=city)
        all_data.append(data)
    
    data = pd.concat(all_data)
    monthly_data = modify_weather_data(data, VARIABLES)
    graphic(monthly_data)

if __name__ == "__main__":
    main()
