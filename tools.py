##############################################
# Description: This file contains the DataTools class which provides various tools for data manipulation and analysis.
# Author: Théo Wey / Thybalt Carratala / Ingrid Mendomo
# Date: 2025-02-21
# Version: 1.0
##############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
from datetime import datetime
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from typing import List, Optional, Tuple
import pandas as pd
import datetime as dt
from data import Data
from tqdm import tqdm


class DataTools:
    @staticmethod
    def open_files_in_directory(
        directory_path: str,
        name: str,
        separator: str,
        time: str = "",
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
        
        if time:
            print(
                f"Opening files containing '{name}' in directory: {directory_path} at {time}"
            )
        else:
            print(
                f"Opening files containing '{name}' in directory: {directory_path}"
            )
        try:
            # Get matching filenames first
            matching_files = [
                filename for filename in os.listdir(directory_path)
                if filename.endswith(file_extension) and name in filename and time in filename
            ]
            
            # Initialize progress bar
            files = []
            for filename in tqdm(matching_files, desc="Loading files"):
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
                    continue
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
        
        print("Renaming headers...")
        for i in tqdm(range(len(data)), desc="Renaming headers"):
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
    def merge_dataframes(
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        key1: str,
        key2: str,
        key3: List[str],
        debug_statement: bool = False,
    ) -> pd.DataFrame:
        """
        Merge two dataframes on the specified keys.
        Args:
            data1 (pd.DataFrame): The first DataFrame to merge.
            data2 (pd.DataFrame): The second DataFrame to merge.
            key1 (str): The key to merge on in the first DataFrame.
            key2 (str): The key to merge on in the second DataFrame.
            key3 (List[str]): The columns to select in the second DataFrame.
        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        
        print("Merging dataframes...")
        # Create a progress bar with steps
        progress = tqdm(total=4, desc="Merge progress")
        
        # Ensure the keys are of the same type and format
        data1[key1] = data1[key1].astype(str)
        progress.update(1)
        
        data2[key2] = data2[key2].astype(str)
        progress.update(1)
        
        selected_columns = [key2] + key3
        merged_data = pd.merge(
            data1,
            data2[selected_columns],
            left_on=key1,
            right_on=key2,
            how="left",
        )
        progress.update(1)
        
        # Check for NaN values in the merged data for key3 columns
        nan_rows = merged_data[merged_data[key3].isna().any(axis=1)]
        if not nan_rows.empty and debug_statement:
            print(f"Found {len(nan_rows)} rows with NaN values in {key3}")

            # Get the key1 values that have NaN in key3 columns
            nan_keys = nan_rows[key1].unique().tolist()
            print(f"Missing data for {key1} values: {nan_keys}")

            # Look up these values in the original data2
            for missing_key in nan_keys:
                # Check if the missing key exists exactly in data2
                exact_matches = data2[data2[key2] == missing_key]
                if not exact_matches.empty:
                    print(
                        f"Found exact matches in data2 for {missing_key}, but merge failed:"
                    )
                    print(exact_matches[selected_columns])

                # Try to find similar matches (case insensitive or partial)
                potential_matches = data2[
                    data2[key2]
                    .astype(str)
                    .str.contains(missing_key, na=False, case=False)
                ]
                if not potential_matches.empty:
                    print(
                        f"Found potential matches in data2 for {missing_key}:"
                    )
                    print(potential_matches[selected_columns])
        
        progress.update(1)
        progress.close()
        print("Merge complete!")
        
        return merged_data

    @staticmethod
    def calul_capacity(data: pd.DataFrame, city: str) -> pd.DataFrame:
        """
        Calculate the capacity of the bike stations based on the total number of bike stands.
        Args:
            data (pd.DataFrame): The DataFrame containing the station data.
        Returns:
            pd.DataFrame: A DataFrame with the capacity of the bike stations.
        """
        try:
            unique_stations = data[data["city"] == city].drop_duplicates(
                subset=["id"]
            )
            capacity = unique_stations["bike_stands"].sum()
            # print(f"Capacity of bike stations in {city}: {capacity}")
            # print(
            #     f"Number of unique stations in {city}: {len(unique_stations)}"
            # )
            # print(
            #     f"Average capacity per station in {city}: {capacity / len(unique_stations)}"
            # )
        except Exception as e:
            print(f"An error occurred: {e}")
            capacity = 0
        return capacity

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
                        station_id = row["id"]  # Station ID
                        places = row["bike_stands"]
                        normalized_data.append([lat, lon, station_id, places])  # Add ID to the list
                    except (ValueError, TypeError):
                        continue

        map_center = [46.603354, 1.888334]

        # Create map
        bike_map = folium.Map(location=map_center, zoom_start=6)
        
        # Add satellite tile layer
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google",
            name="Vue Satellite",
            overlay=False
        ).add_to(bike_map)

        for lat, lon, station_id, places in normalized_data:
            bike_count = DataTools.get_bike_count_at_time(
                bike_data_list, station_id, specified_time
            )
            if bike_count is not None:
                # Convert bike_count to float before division
                bike_count = float(bike_count)
                places_int = int(places)
                percentage = (bike_count / places_int) * 100
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>Station ID: {station_id}</h4>
                    <p>Vélos disponibles: {int(bike_count)}/{places_int}</p>
                    <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
                        <div style="width: {percentage}%; background-color:rgb(18, 38, 36); height: 20px; border-radius: 5px;"></div>
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
            
        # Add layer control
        folium.LayerControl().add_to(bike_map)

        # Save the map to an HTML file
        bike_map.save("velo_map.html")

    @staticmethod
    def calculate_use(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the daily use of bikes based on the number of bikes available.
        Args:
            data (pd.DataFrame): The DataFrame containing the bike data.
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                1. DataFrame with the daily use of bikes (by date)
                2. DataFrame with total bikes used per station
        """
        try:
            if "datetime" not in data.columns:
                print("Error: 'datetime' column not found in the data")
                return pd.DataFrame(
                    columns=["date", "total_bikes_used"]
                ), pd.DataFrame(columns=["id", "used_bikes"])

            # Make a copy to avoid modifying original data
            data_copy = data.copy()
            data_copy["datetime"] = pd.to_datetime(
                data_copy["datetime"], format="%Y-%m-%dT%H:%M", errors="coerce"
            )

            # Drop rows with invalid datetime
            data_copy = data_copy.dropna(subset=["datetime"])

            # Extract date and hour into separate columns
            data_copy["date"] = data_copy["datetime"].dt.date
            data_copy["hour"] = data_copy["datetime"].dt.time

            # Convert bikes column to numeric
            data_copy["bikes"] = pd.to_numeric(
                data_copy["bikes"], errors="coerce"
            )
            data_copy = data_copy.dropna(subset=["bikes"])

            # Group by id and sort by datetime within each group
            data_copy = data_copy.sort_values(["id", "date", "hour"])

            # Calculate the difference in bikes compared to the previous record for each station
            data_copy["bikes_diff"] = data_copy.groupby("id")["bikes"].diff()

            # Negative difference means bikes were taken (more bikes used)
            # We only count newly used bikes (negative differences)
            data_copy["bikes_used"] = data_copy["bikes_diff"].apply(
                lambda x: abs(x) if x < 0 else 0
            )

            # Summarize by date to get daily usage
            daily_use = (
                data_copy.groupby("date")
                .agg(total_bikes_used=("bikes_used", "sum"))
                .reset_index()
            )

            # Get total bikes used per station
            bikes_used = (
                data_copy.groupby("id")["bikes_used"].sum().reset_index()
            )
            bikes_used.columns = ["id", "bikes_used"]

            # get total bike used per hour of the day
            # Convert hour to string format for grouping
            data_copy["hour"] = data_copy["hour"].apply(
                lambda x: x.strftime("%H")
            )

            # Group by both date and hour to get usage per hour per day
            bikes_used_per_hour = (
                data_copy.groupby(["date", "hour"])["bikes_used"]
                .sum()
                .reset_index()
            )
            bikes_used_per_hour.columns = ["date", "hour", "bikes_used"]

            return daily_use, bikes_used, bikes_used_per_hour

        except Exception as e:
            print(f"An error occurred in calculate_use: {e}")
            # Return empty DataFrames with the expected structure
            return pd.DataFrame(
                columns=["date", "total_used_bikes"]
            ), pd.DataFrame(columns=["id", "bikes_used"])
        # upgrade calculate_use to return the use by period : morning, afternoon, evening, night, total use on day, month, year, season (spring, summer, autumn, winter) and the total use of the period

    @staticmethod
    def extract_monthly_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert the provided data to monthly aggregates.
        If a 'date' column is missing but a 'datetime' (or 'time') column exists,
        or if the index is a DatetimeIndex, use it to build a 'date' column.
        """
        if "date" not in data.columns:
            if "datetime" in data.columns:
                data["date"] = pd.to_datetime(data["datetime"], errors="coerce").dt.date
            elif "time" in data.columns:
                data["date"] = pd.to_datetime(data["time"], errors="coerce").dt.date
            elif isinstance(data.index, pd.DatetimeIndex):
                data["date"] = data.index.date
            else:
                raise KeyError("Neither 'date', 'datetime', nor 'time' columns are present in the data.")
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"])
        data_monthly = data.groupby(data["date"].dt.to_period("M")).sum().reset_index()
        data_monthly["date"] = data_monthly["date"].dt.to_timestamp()
        return data_monthly

    @staticmethod
    def plot_usage(
        cities: List[str],
        x: list[pd.DataFrame],
        y: list[pd.DataFrame],
        title: str,
        x_label: str,
        y_label: str,
    ) -> None:
        """
        Plot the usage of bikes over time for two cities."
        """
        plt.figure(figsize=(12, 6))
        plt.plot(x[0], y[0], "b-", label=cities[0])
        plt.plot(x[1], y[1], "r-", label=cities[1])
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def corr_analysis(data: list[pd.DataFrame], columns_to_extract: list) -> None:
        """
        Perform a correlation analysis between two datasets and display the correlation matrix.
        Args:
            data (list[pd.DataFrame]): A list containing two DataFrames
            columns_to_extract (list): A list containing column names to use from each dataset.
                First element can be a string or list of strings for the first dataset.
                Second element can be a string or list of strings for the second dataset.
                e.g. ["total_bikes_used", ["temp", "humidity"]]
        """
        # The first dataset is used as is
        df1 = data[0]
        cols1 = columns_to_extract[0] if isinstance(columns_to_extract[0], list) else [columns_to_extract[0]]
        
        for col in cols1:
            if col not in df1.columns:
                raise KeyError(f"Column '{col}' not found in the first dataset.")
        
        # The second dataset is used as is (no monthly aggregation)
        df2 = data[1]
        cols2 = columns_to_extract[1] if isinstance(columns_to_extract[1], list) else [columns_to_extract[1]]
        
        for col in cols2:
            if col not in df2.columns:
                raise KeyError(f"Column '{col}' not found in the second dataset.")
        
        # Extract and reset indices for both dataframes
        df1_selected = df1[cols1].reset_index(drop=True)
        df2_selected = df2[cols2].reset_index(drop=True)
        
        # Find the minimum length of both dataframes to avoid index mismatches
        min_len = min(len(df1_selected), len(df2_selected))
        df1_selected = df1_selected[:min_len]
        df2_selected = df2_selected[:min_len]
        
        # Concatenate and convert to numeric
        combined = pd.concat([df1_selected, df2_selected], axis=1)
        combined = combined.apply(pd.to_numeric, errors="coerce")
        
        # Compute correlation
        corr_matrix = combined.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def predict_bike_usage(
        usage_data: pd.DataFrame,
        weather_data: pd.DataFrame = None,
        pollution_data: pd.DataFrame = None,
        target_col: str = "total_bikes_used",
        threshold: float = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """
        Predict bike usage based on historical data and optional weather/pollution factors.
        
        Args:
            usage_data (pd.DataFrame): DataFrame with bike usage data (must contain 'date' and target_col)
            weather_data (pd.DataFrame, optional): DataFrame with weather data
            pollution_data (pd.DataFrame, optional): DataFrame with pollution data
            target_col (str): Column name for the bike usage to predict
            threshold (float, optional): Threshold to classify high vs. low usage days
            test_size (float): Proportion of test data for model evaluation
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Results of the analysis including models and performance metrics
        """
        print("Starting bike usage prediction analysis...")
        
        # Prepare the dataset by merging usage with weather and pollution data if provided
        df = usage_data.copy()
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        features = []
        
        # Add weather data if provided
        if isinstance(weather_data, pd.DataFrame) and not weather_data.empty:
            weather_data = weather_data.copy()
            # Check and create date column from various time columns
            if 'request_time' in weather_data.columns:
                weather_data['date'] = pd.to_datetime(weather_data['request_time']).dt.date
            elif 'datetime' in weather_data.columns:
                weather_data['date'] = pd.to_datetime(weather_data['datetime']).dt.date
            elif 'date' in weather_data.columns:
                weather_data['date'] = pd.to_datetime(weather_data['date'])
            else:
                print("Warning: No date column found in weather_data. Skipping weather data.")
                weather_data = None
                
            # Only merge if date column was created successfully
        # Add pollution data if provided
        if isinstance(pollution_data, pd.DataFrame) and not pollution_data.empty:
            pollution_data = pollution_data.copy()
            # Check and create date column from various time columns
            if 'date' not in pollution_data.columns and 'datetime' in pollution_data.columns:
                pollution_data['date'] = pd.to_datetime(pollution_data['datetime']).dt.date
            elif 'date' in pollution_data.columns:
                pollution_data['date'] = pd.to_datetime(pollution_data['date'])
            else:
                print("Warning: No date column found in pollution_data. Skipping pollution data.")
                pollution_data = None
                
            # Only merge if date column was created successfully
            if pollution_data is not None and 'date' in pollution_data.columns:
                # Convert to datetime object (not datetime.date) for proper merging
                pollution_data['date'] = pd.to_datetime(pollution_data['date'])
                df['date'] = pd.to_datetime(df['date'])
                
                # Merge with usage data
                df = pd.merge(df, pollution_data, on='date', how='left')
                features.extend(['NO', 'NO2', 'NOX as NO2', 'O3', 'PM10', 'PM2.5'])
                features = [f for f in features if f in df.columns]
            pollution_data = pollution_data.copy()
            if 'date' not in pollution_data.columns and 'datetime' in pollution_data.columns:
                pollution_data['date'] = pd.to_datetime(pollution_data['datetime']).dt.date
            elif 'date' in pollution_data.columns:
                pollution_data['date'] = pd.to_datetime(pollution_data['date'])
                
            # Merge with usage data
            df = pd.merge(df, pollution_data, on='date', how='left')
            features.extend(['NO', 'NO2', 'NOX as NO2', 'O3', 'PM10', 'PM2.5'])
            features = [f for f in features if f in df.columns]
        
        # Add day of week, month as features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        features.extend(['day_of_week', 'month', 'is_weekend'])
        
        # Drop rows with missing values
        df = df.dropna(subset=[target_col] + features)
        
        if len(df) < 10:
            print(f"Error: Not enough data points after cleaning. Only {len(df)} rows available.")
            return {}
            
        print(f"Dataset prepared with {len(df)} rows and features: {features}")
        
        # Create classification target if threshold provided
        if threshold is not None:
            classification_col = 'high_usage'
            df[classification_col] = False
            df.loc[df[target_col] >= threshold, classification_col] = True
            
            # Print statistics
            percentage_high = (df[classification_col].mean()) * 100
            nb_high = df[classification_col].sum()
            nb_low = len(df) - nb_high
            
            print(f"High usage statistics:")
            print(f"- Percentage of high usage days: {percentage_high:.2f}%")
            print(f"- Number of high usage days: {nb_high}")
            print(f"- Number of low usage days: {nb_low}")
        
        # Create usage categories based on quantiles
        quantiles = df[target_col].quantile([0.33, 0.66])
        cat_col = f"{target_col}_category"
        df[cat_col] = pd.cut(
            df[target_col], 
            bins=(-np.inf, quantiles[0.33], quantiles[0.66], np.inf), 
            labels=['low', 'medium', 'high']
        )
        
        results = {}
        
        # 1. Regression model for predicting exact usage
        try:
            X = df[features]
            y = df[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train regression model
            reg_model = LinearRegression()
            reg_model.fit(X_train_scaled, y_train)
            
            # Evaluate regression model
            y_pred = reg_model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nRegression Model Results:")
            print(f"- Mean Absolute Error: {mae:.2f}")
            print(f"- Root Mean Squared Error: {rmse:.2f}")
            print(f"- R² Score: {r2:.4f}")
            
            results['regression'] = {
                'model': reg_model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'scaler': scaler,
                'features': features
            }
            
            # Visualize regression results
            import matplotlib as mpl
            mpl.rcParams['agg.path.chunksize'] = 10000  # Increase chunk size for plotting

            plt.figure(figsize=(12, 6))
            # Use the test indices to get corresponding dates from the original dataframe
            test_indices = y_test.index
            test_dates = df.iloc[test_indices]['date']

            # If there are too many points, sample them for plotting
            if len(test_dates) > 1000:
                sample_size = min(1000, len(test_dates))
                sample_indices = np.random.choice(len(test_dates), sample_size, replace=False)
                sample_dates = test_dates.iloc[sample_indices]
                sample_y_test = y_test.iloc[sample_indices]
                sample_y_pred = y_pred[sample_indices]
                plt.scatter(sample_dates, sample_y_test, label='Actual Usage', marker='o')
                plt.scatter(sample_dates, sample_y_pred, label='Predicted Usage', marker='x')
            else:
                plt.scatter(test_dates, y_test, label='Actual Usage', marker='o')
                plt.scatter(test_dates, y_pred, label='Predicted Usage', marker='x')
                
            plt.xlabel('Date/Time')
            plt.ylabel('Usage')
            plt.title('Actual vs Predicted Usage Over Time')
            plt.legend()
            plt.tight_layout()
            plt.show()
            
            # 2. Classification model if threshold provided
            if threshold is not None:
                X = df[features]
                y = df[classification_col]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train classification model
                log_model = LogisticRegression(max_iter=1000, random_state=random_state)
                log_model.fit(X_train_scaled, y_train)
                
                # Evaluate classification model
                y_pred = log_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                
                print(f"\nBinary Classification Model (High vs. Low Usage):")
                print(f"- Accuracy: {accuracy:.2f}")
                print("- Confusion Matrix:")
                print(conf_matrix)
                
                results['classification'] = {
                    'model': log_model,
                    'accuracy': accuracy,
                    'confusion_matrix': conf_matrix,
                    'classification_report': class_report
                }
                
                # Plot confusion matrix
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['Low Usage', 'High Usage'],
                            yticklabels=['Low Usage', 'High Usage'])
                plt.title('Confusion Matrix for Usage Classification')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.tight_layout()
                plt.show()
                
            # 3. Category prediction model (low, medium, high)
            X = df[features]
            y = df[cat_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train KNN model
            knn_model = KNeighborsClassifier(n_neighbors=5)
            knn_model.fit(X_train_scaled, y_train)
            
            # Evaluate KNN model
            y_pred = knn_model.predict(X_test_scaled)
            cat_accuracy = accuracy_score(y_test, y_pred)
            cat_conf_matrix = confusion_matrix(y_test, y_pred)
            cat_report = classification_report(y_test, y_pred)
            
            print(f"\nCategory Classification Model (Low/Medium/High):")
            print(f"- Accuracy: {cat_accuracy:.2f}")
            print("- Confusion Matrix:")
            print(cat_conf_matrix)
            
            results['category_classification'] = {
                'model': knn_model,
                'accuracy': cat_accuracy,
                'confusion_matrix': cat_conf_matrix,
                'classification_report': cat_report
            }
            
            # Plot category confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cat_conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Low', 'Medium', 'High'],
                        yticklabels=['Low', 'Medium', 'High'])
            plt.title('Confusion Matrix for Usage Category Classification')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.show()
            
            # Time series plot of actual vs predicted
            plt.figure(figsize=(14, 6))
            test_dates = df.iloc[y_test.index]['date']
            plt.plot(test_dates, y_test.values, label='Actual Usage', marker='o', alpha=0.5)
            plt.plot(test_dates, y_pred, label='Predicted Usage', marker='x', alpha=0.5)
            plt.title('Time Series of Actual vs Predicted Bike Usage')
            plt.xlabel('Date')
            plt.ylabel('Bike Usage')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            
            return results
            
        except Exception as e:
            print(f"Error during model training/evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
        
    @staticmethod
    def get_api_data(url, api_key):
        headers = {"Accept": "application/json"}
        params = {"apiKey": api_key}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête : {e}")
            return None
        
    @staticmethod
    def get_station_names(api_key, contract_name):
        url = "https://api.jcdecaux.com/vls/v3/stations"
        headers = {"Accept": "application/json"}
        params = {"apiKey": api_key, "contract": contract_name}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            stations = response.json()
            station_data = {station["number"]: station["name"] for station in stations}
            print(f"Stations pour {contract_name}: {station_data}")
            return station_data
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête pour {contract_name}: {e}")
            return None

    @staticmethod
    def get_station_details(api_key, contract_name, station_number):
        url = f"https://api.jcdecaux.com/vls/v3/stations/{station_number}"
        headers = {"Accept": "application/json"}
        params = {"apiKey": api_key, "contract": contract_name}

        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            station_info = response.json()
            return station_info
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête pour la station {station_number} de {contract_name}: {e}")
            return None
    @staticmethod
    def save_station_data_to_xlsx(station_data, contract_name, api_key, date_str):
        # Nom du fichier
        filename = f"databike_{contract_name}_{date_str}.xlsx"

        # Créer le dossier "real_data" s'il n'existe pas
        os.makedirs("real_data", exist_ok=True)

        # Chemin complet du fichier
        filepath = os.path.join("real_data", filename)

        # Liste des colonnes attendues
        headers = [
            "station_number", "station_name", "available_bikes", 
            "available_bike_stands", "mechanical_bikes", "electrical_bikes",
            "latitude", "longitude", "status", "last_update", "address", "banking", 
            "bonus", "connected", "overflow"
        ]

        # Liste pour stocker les données
        data = []

        for station_number, station_name in station_data.items():
            # Récupérer les détails de chaque station
            station_details = DataTools.get_station_details(api_key, contract_name, station_number)
            if station_details:
                row = [
                    station_details.get("number", ""),
                    station_details.get("name", ""),
                    station_details["totalStands"]["availabilities"].get("bikes", 0),
                    station_details["totalStands"]["availabilities"].get("stands", 0),
                    station_details["totalStands"]["availabilities"].get("mechanicalBikes", 0),
                    station_details["totalStands"]["availabilities"].get("electricalBikes", 0),
                    station_details["position"].get("latitude", ""),
                    station_details["position"].get("longitude", ""),
                    station_details.get("status", ""),
                    station_details.get("lastUpdate", ""),
                    station_details.get("address", ""),
                    station_details.get("banking", ""),
                    station_details.get("bonus", ""),
                    station_details.get("connected", ""),
                    station_details.get("overflow", "")
                ]
                data.append(row)
                data.append(row)

        # Création d'un DataFrame pandas
        df = pd.DataFrame(data, columns=headers)

        # Sauvegarde en fichier Excel
        df.to_excel(filepath, index=False)

        print(f"Données sauvegardées dans le fichier {filepath}")
