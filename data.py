##############################################
# Description: This file contains the Data class which is used to store and clean data.
# Author: Théo Wey / Thybalt Carratala / Ingrid Mendomo
# Date: 2025-02-21
# Version: 1.0
##############################################

import pandas as pd


class Data:
    def __init__(self):
        self.data = pd.DataFrame()
        self.data_numerique = []

    def __del__(self):
        del self.data
        del self.data_numerique

    def get_data(self, data):
        self.data = pd.DataFrame()
        try:
            chunks = []
            for d in data:
                chunk = pd.DataFrame(d)
                chunks.append(chunk)
                if len(chunks) >= 10:  # Process in chunks of 10
                    self.data = pd.concat(
                        [self.data] + chunks, ignore_index=True
                    )
                    chunks = []
            if chunks:
                self.data = pd.concat([self.data] + chunks, ignore_index=True)
            self.__clean_data()

        except ValueError:
            print("Data could not be concatenated.")
        except TypeError:
            print("Data could not be concatenated.")
        except MemoryError as e:
            print(f"Not enough memory to concatenate data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def __clean_data(self):
        # Remove duplicates
        # Keep rows with unique combinations of all columns except 'id'
        # Handle missing values
        self.data.drop_duplicates()
        self.data = self.data.fillna(
            0
        )  # Fill missing values with 0, you can change this to a more appropriate value or method
        # Drop columns with specific names
        values_to_drop = [
            "request_time",
            "lon",
            "lat",
            "weather_id",
            "weather_ma",
        ]
        # Drop columns that match the values_to_drop list
        columns_to_drop = [
            col for col in self.data.columns if col in values_to_drop
        ]
        if columns_to_drop:
            self.data = self.data.drop(columns=columns_to_drop)
        # Handle outliers (example: removing rows where temperature is outside a reasonable range)
        if "temp" in self.data.columns:
            self.data["temp"] = pd.to_numeric(
                self.data["temp"], errors="coerce"
            )
            self.data = self.data[
                (self.data["temp"] >= -50) & (self.data["temp"] <= 50)
            ]

    def filter_dataframes(self, filter_column, filter_values):
        if filter_column in self.data.columns:
            # Filtrer les lignes selon les valeurs spécifiées
            self.data = self.data[self.data[filter_column].isin(filter_values)]
        self.data_numerique = self.data.select_dtypes(include=["number"])

    def print_data(self, city: str = ""):
        if city:
            print(f"Data for {city}:")
            print(self.data[self.data["city"] == city])
        else:
            print(self.data)
