from tools import DataTools
from data import Data
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Demander à l'utilisateur de spécifier l'heure (exemple: "2022-12-25 14:00")
    specified_time = "2022-10-09 19:58"
    specified_time = datetime.strptime(specified_time, "%Y-%m-%d %H:%M")
    formatted_time = specified_time.strftime("%Y-%m")

    # Chemin du répertoire
    path = r"C:\Users\weyth\Downloads\Python-20250115"
    data_directory = r"C:\Users\weyth\Downloads\Python-20250115\data"
    correct_header_data_bikes = [
        "city",
        "id",
        "request_date",
        "datetime",
        "bikes",
    ]

    # lecture des fichiers de données
    station_data = DataTools.open_files_in_directory(path, "station", "\t")
    weather_data = DataTools.open_files_in_directory(
        data_directory, f"weather_{formatted_time}", ","
    )
    pollution_data = DataTools.open_files_in_directory(
        data_directory, f"pollution_{formatted_time}", ","
    )

    bike_data = DataTools.open_files_in_directory(
        data_directory, f"bike_{formatted_time}", "\t"
    )

    # stocker les données dans un objet Data
    print("data_station")
    data_station = Data()
    data_station.get_data(station_data)
    data_station.filter_dataframes("city", ["amiens", "marseille"])
    del station_data

    print("data_weather")
    data_weather = Data()
    data_weather.get_data(weather_data)
    data_weather.filter_dataframes("name", ["Amiens"])
    del weather_data

    print("data_pollution")
    data_pollution = Data()
    data_pollution.get_data(pollution_data)
    data_pollution.filter_dataframes("name", ["Amiens"])
    del pollution_data

    print("data_bike")
    data_bike = Data()
    bike_data = DataTools.rename_header(
        data=bike_data,
        correct_header=correct_header_data_bikes,
        keep_old_header=True,
    )
    data_bike.get_data(bike_data)
    data_bike.filter_dataframes("city", ["amiens", "marseille"])
    del bike_data

    nbr_velo_amiens = 0  # Initialize the variable

    if not data_station.data.empty:
        if (
            "bike_stands" in data_station.data.columns
            and "amiens" in data_station.data["city"].values
        ):
            nbr_velo_amiens = data_station.data[
                data_station.data["city"] == "amiens"
            ]["bike_stands"].sum()
            print(f"nombre de velo amiens: {nbr_velo_amiens}")
        else:
            print(
                "Les données de 'bike_stands' pour 'amiens' ne sont pas disponibles."
            )

    velo = DataTools.calculate_daily_bike_use(
        data_bike.data,
        periode=formatted_time,
        city="amiens",
        total_bikes=nbr_velo_amiens,
    )

    print(data_pollution.data)
    # Appel de la fonction corr_analysis
    DataTools.corr_analysis(
        [velo, data_weather.data],
        {
            "data1": "total_bikes",
            "data2": [
                "temp",
                "feels_like",
                "temp_min",
                "temp_max",
                "pressure",
                "humidity",
                "sea_level",
                "grnd_level",
                "speed",
                "deg",
                "gust",
                "clouds",
            ],
        },
    )
    DataTools.corr_analysis(
        [velo, data_pollution.data],
        {
            "data1": "total_bikes",
            "data2": [
                "PM10",
                "PM2.5",
                "O3",
                "NO2",
                "SO2",
                "CO",
                "NO",
                "NOX as NO2",
            ],
        },
    )
    plt.show()


if __name__ == "__main__":
    main()
