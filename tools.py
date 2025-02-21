import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress
from typing import List, Optional, Tuple
import pandas as pd
import datetime as dt
from data import Data


class DataTools:
    def __init__(self):
        pass

    @staticmethod
    def open_files_in_directory(
        directory_path: str,
        name: str,
        separator: str,
        file_extension: str = ".txt",
    ) -> List[pd.DataFrame]:
        """
        Open and read files in the specified directory that match the given name and file extension.

        Args:
            directory_path (str): The path to the directory containing the files.
            name (str): The name pattern to match files.
            separator (str): The separator used in the files.
            file_extension (str, optional): The file extension to match. Defaults to ".txt".

        Returns:
            List[pd.DataFrame]: A list of pandas DataFrames containing the data from the matched files.
        """
        print(
            f"Opening files containing '{name}' in directory: {directory_path}"
        )
        try:
            # Retrieve files in the directory
            files = []
            for filename in os.listdir(directory_path):
                if filename.endswith(file_extension) and name in filename:
                    try:
                        file = pd.read_csv(
                            os.path.join(directory_path, filename),
                            sep=separator,
                            encoding="utf-8",
                            on_bad_lines="skip",
                        )
                        string_columns = file.select_dtypes(["object"]).columns
                        file[string_columns] = (
                            file[string_columns]
                            .astype(str)
                            .apply(lambda x: x.str.replace(",", "."))
                        )
                        files.append(file)
                    except pd.errors.ParserError:
                        print(f"Error parsing file: {filename}")
                    except pd.errors.EmptyDataError:
                        print(f"No columns to parse from file: {filename}")
        except FileNotFoundError:
            print(f"Directory not found: {directory_path}")
            return []
        except Exception as e:
            print(f"Error opening files in directory: {e}")
            return []

        return files

    @staticmethod
    def rename_header(data: list, correct_header, keep_old_header=False):
        """
        Rename the headers of the dataframes in the list data to the correct header.
        Args:
            data (list): A list of dataframes to process.
            correct_header (list): The correct header to replace the existing one.
            keep_old_header (bool): Whether to keep the old header as a new row in the dataframe.
        Returns:
            list: A list of dataframes with the correct header.
        """
        for i in range(len(data)):
            # Copy the current header of all_data_bike before replacing it
            current_headers = [data[i].columns.tolist()]
            # Replace the header of all_data_bike with the correct one
            data[i] = data[i].rename(
                columns=dict(zip(data[i].columns, correct_header))
            )
            # Add the current headers as a new row in each dataframe
            if keep_old_header:
                new_row = pd.DataFrame(
                    [current_headers[0]], columns=correct_header
                )
                data[i] = pd.concat([new_row, data[i]], ignore_index=True)
        print("Headers replacement done.")
        return data

    @staticmethod
    def get_bike_count_at_time(
        bike_data_list: List[pd.DataFrame], station_id: int, time: datetime
    ) -> Optional[int]:
        """
        Get the number of bikes available at a specific station and time.
        Args:
            bike_data_list (List[pd.DataFrame]): A list of DataFrames containing bike data.
            station_id (int): The ID of the station to check.
            time (datetime): The specific time to check.
        Returns:
            Optional[int]: The number of bikes available at the station and time, or None if the data is not available.
        """
        for bike_data in bike_data_list:
            if "datetime" in bike_data.columns:
                bike_data["datetime"] = pd.to_datetime(
                    bike_data["datetime"], errors="coerce"
                )
            else:
                continue
            station_data = bike_data[
                (bike_data["id"] == station_id)
                & (bike_data["datetime"] == time)
            ]
            if not station_data.empty:
                return station_data.iloc[0]["bikes"]
        return None

    @staticmethod
    def create_bike_station_map(
        station_data_list: List[pd.DataFrame],
        bike_data_list: List[pd.DataFrame],
        specified_time: datetime,
    ) -> None:
        """
        Create a map showing the bike stations with the number of available bikes at a specific time.
        Args:
            station_data_list (List[pd.DataFrame]): A list of DataFrames containing station data.
            bike_data_list (List[pd.DataFrame]): A list of DataFrames containing bike data.
            specified_time (datetime): The specific time to check for available bikes.
        """
        normalized_data = []
        for df in station_data_list:
            if all(col in df.columns for col in ["longitude", "latitude"]):
                df = df.dropna(subset=["longitude", "latitude"])
                for _, row in df.iterrows():
                    try:
                        lat = float(str(row["latitude"]).replace(",", "."))
                        lon = float(str(row["longitude"]).replace(",", "."))
                        station_id = row["id"]
                        places = row["bike_stands"]
                        normalized_data.append([lat, lon, station_id, places])
                    except (ValueError, TypeError):
                        continue

        map_center = [46.603354, 1.888334]
        bike_map = folium.Map(location=map_center, zoom_start=6)

        for lat, lon, station_id, places in normalized_data:
            bike_count = DataTools.get_bike_count_at_time(
                bike_data_list, station_id, specified_time
            )
            if bike_count is not None:
                percentage = (bike_count / int(places)) * 100
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>Station ID: {station_id}</h4>
                    <p>Vélos disponibles: {bike_count}/{int(places)}</p>
                    <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
                        <div style="width: {percentage}%; background-color: #76c7c0; height: 20px; border-radius: 5px;"></div>
                    </div>
                </div>
                """
                color = (
                    "green"
                    if bike_count > 5
                    else "orange" if bike_count > 0 else "red"
                )
            else:
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>Station ID: {station_id}</h4>
                    <p>Aucune donnée disponible</p>
                </div>
                """
                color = "gray"

            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(icon="bicycle", prefix="fa", color=color),
            ).add_to(bike_map)

        bike_map.save("velo_map.html")

    @staticmethod
    def calculate_daily_bike_use(
        data: pd.DataFrame, periode: str, city: str, total_bikes: int
    ) -> pd.DataFrame:
        """
        Calculate the daily total number of bikes at 12h for a given city and period.
        Args:
            data (pd.DataFrame): A DataFrame containing bike data with a 'datetime' column.
            periode (datetime): The period (year and month) for which to calculate the daily totals.
            city (str): The name of the city for which to calculate the daily totals.
            total_bikes (int): The total number of bikes to be used in the calculation.
        Returns:
            pd.DataFrame: A DataFrame containing the daily total number of bikes at 12h for each day in the given period.
                          The DataFrame has columns: 'city', 'date', and 'total_bikes'.
        """

        year = int(periode.split("-")[0])
        month = int(periode.split("-")[1])

        all_data_bike = [data] if isinstance(data, pd.DataFrame) else data

        filtered_data_bike = [
            df[df["city"] == city]
            for df in all_data_bike
            if "city" in df.columns
        ]

        daily_totals = []

        for day in range(1, 32):
            try:
                current_date = datetime(year, month, day).date()
                target_time = datetime(year, month, day, 12, 0, 0)

                day_files = []
                for df in filtered_data_bike:
                    if "datetime" in df.columns:
                        df = df.copy()
                        df["datetime"] = pd.to_datetime(
                            df["datetime"], errors="coerce"
                        )
                        day_files.append(df[df["datetime"] == target_time])

                day_files = [df for df in day_files if not df.empty]
                if day_files:
                    daily_data = pd.concat(day_files, ignore_index=True)

                if "bikes" in daily_data.columns:
                    daily_data["bikes"] = pd.to_numeric(
                        daily_data["bikes"], errors="coerce"
                    )
                    total_bikes_at_12h = daily_data["bikes"].sum()
                    daily_totals.append(
                        {
                            "city": city,
                            "date": current_date,
                            "total_bikes": total_bikes - total_bikes_at_12h,
                        }
                    )
                else:
                    daily_totals.append(
                        {
                            "city": city,
                            "date": current_date,
                            "total_bikes": total_bikes,
                        }
                    )

            except ValueError:
                continue

        if daily_totals:
            combined_daily_totals = pd.DataFrame(daily_totals)
        else:
            combined_daily_totals = pd.DataFrame(
                columns=["city", "date", "total_bikes"]
            )

        return combined_daily_totals

    @staticmethod
    def extract_monthly_data(data: Data, hour: int = 12):
        """
        Extracts and combines all weather files for a given month,
        selects a single row per day at a given hour,
        and filters by specified city (Amiens or Marseille).

        Args:
            data (Data): Instance of the Data class containing weather data.
            periode (str): Period to extract (format "YYYY-MM").
            city (str): City to extract (Amiens or Marseille).
            hour (int): Hour to fix for extraction (default is 12 PM).

        Returns:
            pd.DataFrame: Combined DataFrame with a single row per day for the specified city and hour.
        """
        # Vérifier si les colonnes nécessaires sont présentes
        if not hasattr(data, "columns") or not any(
            col in data.columns for col in ["request_time", "date"]
        ):
            print(
                "Missing 'request_time' or 'date' column in the weather data."
            )
            return pd.DataFrame()

        # Convertir le timestamp en datetime pour faciliter le filtrage
        if "request_time" in data.columns:
            data["request_time"] = pd.to_datetime(
                data["request_time"], errors="coerce"
            )
            data["date"] = data["request_time"].dt.date
            data["hour"] = data["request_time"].dt.hour
        elif "date" in data.columns:
            data["date"] = pd.to_datetime(data["date"], errors="coerce")
            data["hour"] = data["date"].dt.hour

        # Filtrer pour l'heure spécifiée
        daily_data_at_date = data[data["hour"] == hour]

        # Garder une seule ligne par jour
        final_data = daily_data_at_date.groupby("date").first().reset_index()

        # Supprimer les colonnes inutiles si nécessaire
        final_data = final_data.drop(columns=["hour"], errors="ignore")

        return final_data

    @staticmethod
    def corr_analysis(*args: any):
        """
        Perform a correlation analysis between two datasets and display the correlation matrix.
        Args:
            *args: Variable length argument list containing the datasets and columns to extract.
        """
        datas, columns_to_extract = args

        # Extraire la colonne 'total_bikes' de 'velo'
        data_1 = datas[0][columns_to_extract["data1"]]
        # Reshaper la colonne pour s'assurer que l'index est bien aligné
        data_1 = data_1.reset_index(drop=True)

        data_2 = DataTools.extract_monthly_data(data=datas[1])
        data_2 = data_2[columns_to_extract["data2"]]
        corr_analys = pd.concat([data_1, data_2], axis=1)
        corr_analys = corr_analys.apply(pd.to_numeric, errors="coerce")
        df_numerique = corr_analys.select_dtypes(include=["float64", "int64"])

        # Calculer la matrice de corrélation
        corr_matrix = df_numerique.corr()

        # Afficher la matrice de corrélation avec seaborn
        plt.figure(figsize=(10, 8))  # Taille de la figure
        sns.heatmap(
            corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
        )
        plt.title("Matrice de Corrélation")
