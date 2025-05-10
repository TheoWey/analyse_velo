from tools import DataTools
from data import Data
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import traceback


def main():
    # Ask the user to specify the time (example: "2022-12-25 14:00:00")
    specified_time = "2022-11-29 10:00"
    specified_time = datetime.strptime(specified_time, "%Y-%m-%d %H:%M")
    formatted_time = specified_time.strftime("%Y")

    # Directory path
    path = r"..\data_files"
    data_directory = r"..\data_files\data"
    correct_header_data_bikes = [
        "city",
        "id",
        "request_date",
        "datetime",
        "bikes",
    ]

    # Store data in a Data object
    # read station data
    station_data = DataTools.open_files_in_directory(
        path, f"bike_station", "\t"
    )
    bike_station = Data()
    bike_station.get_data(station_data)
    bike_station.filter_dataframes("city", ["amiens", "marseille"])
    del station_data

    # read pollution station data
    station_data = DataTools.open_files_in_directory(
        path, "pollution_station", ","
    )
    pollution_station = Data()
    pollution_station.get_data(station_data)
    pollution_station.filter_dataframes("city", ["amiens", "marseille"])
    del station_data

    # read weather data
    weather_data = DataTools.open_files_in_directory(
        data_directory, f"weather_{formatted_time}", ","
    )
    data_weather = Data()
    data_weather.get_data(weather_data)
    data_weather.filter_dataframes("name", ["Amiens", "Marseille"])
    del weather_data

    # read pollution data
    pollution_data = DataTools.open_files_in_directory(
        data_directory, f"pollution_{formatted_time}", ","
    )
    data_pollution = Data()
    data_pollution.get_data(pollution_data)
    data_pollution.filter_dataframes("name", ["Amiens"])
    del pollution_data

    # read bike data
    bike_data = DataTools.open_files_in_directory(
        data_directory,
        f"bike_{formatted_time}",
        "\t",
    )
    data_bike = Data()
    bike_data = DataTools.rename_header(
        data=bike_data,
        correct_header=correct_header_data_bikes,
        keep_old_header=True,
    )
    data_bike.get_data(bike_data)
    data_bike.filter_dataframes("city", ["amiens", "marseille"])
    del bike_data

    # put bike_stand columns from bike_station in data_bike
    data_bike.data = DataTools.merge_dataframes(
        data_bike.data,
        bike_station.data,
        "id",
        "id",
        ["bike_stands", "latitude", "longitude", "id_pollution"],
    )

    # put pollution columns from pollution_station in data_bike
    data_pollution.data = DataTools.merge_dataframes(
        data_pollution.data,
        pollution_station.data,
        "id",
        "id",
        ["latitude", "longitude"],
    )

    # free memory after use of the station
    del pollution_station
    del bike_station

        # Calculate the number of bikes available in Amiens and Marseille
    bike_count_amiens = DataTools.calul_capacity(data_bike.data, "amiens")
    bike_count_marseille = DataTools.calul_capacity(data_bike.data, "marseille")

    print(f"Number of slots in Amiens: {bike_count_amiens}")
    print(f"Number of slots in Marseille: {bike_count_marseille}")

    dailyuse_amiens, period_use_amiens, useperhour_amiens = (
        DataTools.calculate_use(
            data_bike.data[data_bike.data["city"] == "amiens"]
        )
    )
    dailyuse_marseille, period_use_marseille, useperhour_marseille = (
        DataTools.calculate_use(
            data_bike.data[data_bike.data["city"] == "marseille"]
        )
    )



    # Correlation analysis for dailyuse_amiens
    DataTools.corr_analysis(
        [dailyuse_amiens, data_weather.data],
        ["total_bikes_used", ["temp", "temp_max", "temp_min", "humidity", "speed", "clouds"]],
    )

    DataTools.corr_analysis(
        [dailyuse_amiens, data_pollution.data],
        ["total_bikes_used", ["NO", "NO2", "NOX as NO2", "O3", "PM10", "PM2.5"]],
    )

    # Correlation analysis for dailyuse_marseille
    DataTools.corr_analysis(
        [dailyuse_marseille, data_weather.data],
        ["total_bikes_used", ["temp", "temp_max", "temp_min", "humidity", "speed", "clouds"]],
    )

    DataTools.corr_analysis(
        [dailyuse_marseille, data_pollution.data],
        ["total_bikes_used", ["NO", "NO2", "NOX as NO2", "O3", "PM10", "PM2.5"]],
    )

if __name__ == "__main__":
    try:
        main()
        # Free memory
        gc.collect()
        print("Program completed successfully. Memory freed.")
    except Exception as e:
        print(f"An error occurred: {e}")
        gc.collect()
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        gc.collect()
