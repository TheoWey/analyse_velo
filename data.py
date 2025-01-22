import pandas as pd


class Data:
    def __init__(self):
        self.data = pd.DataFrame()
        self.data_numerique = []

    def get_data(self, data):
        self.data = pd.DataFrame()
        try:
            chunks = [pd.DataFrame(d) for d in data]
            if chunks:
                self.data = pd.concat(chunks, ignore_index=True)
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
        print("Cleaning data...")
        # Remove duplicates
        self.data = self.data.drop_duplicates()
        # Handle missing values
        self.data = self.data.fillna(
            0
        )  # Fill missing values with 0, you can change this to a more appropriate value or method
        # Drop rows containing specific values
        values_to_drop = [
            "request_time",
            "lon",
            "lat",
            "weather_id",
            "weather_ma",
        ]
        pattern = "|".join(values_to_drop)
        self.data = self.data[
            ~self.data.astype(str)
            .apply(lambda x: x.str.contains(pattern))
            .any(axis=1)
        ]
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

    def print_data(self):
        print(self.data)
