import pandas as pd

class data:
    def __init__(self, path, separator='\t'):
        self.__data = [self.__read_data(path, separator)]
        self.__cleaned_data = [self.__clean_data(df) for df in self.__data]
        self.data = self.__cleaned_data

    def __read_data(self, path, separator):
        try:
            # Lecture du fichier avec gestion des exceptions
            print(f"Reading file: {path}")
            data = pd.read_csv(filepath_or_buffer=path, sep=separator)
        except pd.errors.EmptyDataError:
            print(f"No columns to parse from file: {path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return pd.DataFrame()
        
        # Ensure the data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        
        return data

    def __clean_data(self, df):
        # Remove duplicates
        df = df.drop_duplicates()
        # Handle missing values
        df = df.fillna(0)  # Fill missing values with 0, you can change this to a more appropriate value or method
        # Drop rows containing specific values
        values_to_drop = ['request_time', 'lon', 'lat', 'weather_id', 'weather_ma']
        df = df[~df.apply(lambda row: row.astype(str).str.contains('|'.join(values_to_drop)).any(), axis=1)]
        # Handle outliers (example: removing rows where temperature is outside a reasonable range)
        if 'temp' in df.columns:
            df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
            df = df[(df['temp'] >= -50) & (df['temp'] <= 50)]
        return df

    def replace_headers_and_add_original(self, correct_header):
        # Copy the current header of all_data_bike before replacing it
        current_headers = [df.columns.tolist() for df in self.__cleaned_data]
        # Replace the header of all_data_bike with the correct one
        self.__cleaned_data = [df.rename(columns=dict(zip(df.columns, correct_header))) for df in self.__cleaned_data]
        # Add the current headers as a new row in each dataframe
        for i, df in enumerate(self.__cleaned_data):
            new_row = pd.DataFrame([current_headers[i]], columns=correct_header)
            self.__cleaned_data[i] = pd.concat([new_row, df], ignore_index=True)
        self.data = self.__cleaned_data
        print("Headers replacement done.")

    def filter_dataframes(self, filter_column, filter_values):
        filtered_data = []
        for df in self.__cleaned_data:
            if filter_column in df.columns:
                # Filtrer les lignes selon les valeurs spécifiées
                filtered_df = df[df[filter_column].isin(filter_values)]
                filtered_data.append(filtered_df)
        self.data = filtered_data
        del self.__cleaned_data
        del self.__data
        # debug
        # self.print_data
    
    def print_data(self):
        for i, df in enumerate(self.data):
            print(f"Dataframe {i}:")
            print(df)