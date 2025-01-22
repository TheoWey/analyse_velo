from tools import DataTools
from data import Data
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import gc


def main():
    # Ask the user to specify the time (example: "2022-12-25 14:00:00")
    specified_time = "2022-06-02 10:00"
    specified_time = datetime.strptime(specified_time, "%Y-%m-%d %H:%M")
    formatted_time = specified_time.strftime("%Y-%m")

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
    data_weather.filter_dataframes("name", ["Amiens"])
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
    bike_count_marseille = DataTools.calul_capacity(
        data_bike.data, "marseille"
    )

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

    # print(f"\nHourly use in Amiens: \n{useperhour_marseille}")
    # print(f"\nDaily use in Amiens: \n{dailyuse_amiens}")
    # print(f"Bike use on the period {formatted_time} : \n{period_use_amiens}")

    # print(f"\nDaily use in Marseille: \n{dailyuse_marseille}")
    # print(f"Bike use on the period {formatted_time} : \n{period_use_marseille}")

    # Plot daily use for both cities
    DataTools.plot_usage(
        [dailyuse_amiens["date"], dailyuse_marseille["date"]],
        [dailyuse_amiens["used_bikes"], dailyuse_marseille["used_bikes"]],
        ["Amiens", "Marseille"],
        "Daily Bike Usage Comparison",
        "Date",
        "Number of Bikes Used",
    )

    # Plot hourly use for a single day
    # Choose a specific date from the dataset
    # Make sure date column is a datetime type
    if not pd.api.types.is_datetime64_any_dtype(useperhour_amiens["date"]):
        useperhour_amiens["date"] = pd.to_datetime(useperhour_amiens["date"])
    if not pd.api.types.is_datetime64_any_dtype(useperhour_marseille["date"]):
        useperhour_marseille["date"] = pd.to_datetime(
            useperhour_marseille["date"]
        )

    specific_date = useperhour_amiens["date"].dt.date.iloc[
        0
    ]  # Use the first date in the dataset

    # Filter data for the specific date
    amiens_day = useperhour_amiens[
        useperhour_amiens["date"].dt.date == specific_date
    ]
    marseille_day = useperhour_marseille[
        useperhour_marseille["date"].dt.date == specific_date
    ]

    DataTools.plot_usage(
        [amiens_day["hour"], marseille_day["hour"]],
        [amiens_day["used_bikes"], marseille_day["used_bikes"]],
        ["Amiens", "Marseille"],
        f"Hourly Bike Usage Comparison ({specific_date})",
        "Hour",
        "Number of Bikes Used",
    )


if __name__ == "__main__":
    try:
        main()
        # Free memory
        gc.collect()
        print("Program completed successfully. Memory freed.")
    except Exception as e:
        print(f"An error occurred: {e}")
