import numpy as np
import pandas as pd
import os
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from datetime import datetime

def open_file(path, separator):
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
    return data

def open_all_files_in_directory(directory_path, name, separator):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") and name in filename:
            file_path = os.path.join(directory_path, filename)
            data = open_file(file_path, separator)
            all_data.append(data)
    return all_data

def replace_headers_and_add_original(all_data, correct_header):
    # Copy the current header of all_data_bikes before replacing it
    current_headers = [df.columns.tolist() for df in all_data]

    # Replace the header of all_data_bikes with the correct one
    all_data = [df.rename(columns=dict(zip(df.columns, correct_header))) for df in all_data]

    # Add the current headers as a new row in each dataframe
    for i, df in enumerate(all_data):
        new_row = pd.DataFrame([current_headers[i]], columns=correct_header)
        all_data[i] = pd.concat([new_row, df], ignore_index=True)
    
    return all_data


def filter_dataframes(dataframes, filter_column, filter_values):
    filtered_data_weather = []
    for df in dataframes:
        if filter_column in df.columns:
            # Filtrer les lignes selon les valeurs spécifiées
            filtered_df = df[df[filter_column].isin(filter_values)]
            filtered_data_weather.append(filtered_df)
    return filtered_data_weather

def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.dropna()  # or use df.fillna() to fill missing values
    
    # Handle outliers (example: removing rows where temperature is outside a reasonable range)
    if 'temp' in df.columns:
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce')
        df = df[(df['temp'] >= -50) & (df['temp'] <= 50)]
    
    return df

def univariate_analysis(df, column):
    return df[column].describe()

def bivariate_analysis(df, column1, column2):
    return df[[column1, column2]].describe()

def correlation_analysis(df1, df2, column1, column2):
    merged_df = pd.merge(df1, df2, on='timestamp')
    correlation = merged_df[[column1, column2]].corr()
    return correlation

def analyze_bike_data(df):
    # Example: Group by hour to analyze peak/off-peak hours
    df['hour'] = df['timestamp'].dt.hour
    peak_hours = df.groupby('hour').size()
    
    # Example: Group by day of the week to analyze weekdays/weekends
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    weekdays_vs_weekends = df.groupby('day_of_week').size()
    
    return peak_hours, weekdays_vs_weekends

def project_bike_usage(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

# Demander à l'utilisateur de spécifier l'heure (exemple: "2022-12-25 14:00:00")
specified_time = "2022-06-02 20:00:00"
specified_time = datetime.strptime(specified_time, '%Y-%m-%d %H:%M:%S')
formatted_time = specified_time.strftime('%Y-%m-%dT%H')
print(formatted_time)


# Chemin du répertoire
path = r"C:\Users\ingri\Documents\projectdata"
data_directory = r"C:\Users\ingri\Documents\projectdata"
correct_header_data_bikes = ['city', 'station_id', 'request_date', 'answer_date', 'bike_available']

# lecture des fichiers de données
bike_stations = open_file(os.path.join(path, "bike_station.txt"), "\t")
all_data_weather = open_all_files_in_directory(data_directory, f"data_weather_{formatted_time}", ',')
all_data_bikes = open_all_files_in_directory(data_directory, f"data_bike_{formatted_time}", '\t')

# Replace headers and add original headers as a new row
all_data_bikes = replace_headers_and_add_original(all_data_bikes, correct_header_data_bikes)

# suppression des doublons
bike_stations = bike_stations.drop_duplicates()
filtered_data_bikes = [df.drop_duplicates() for df in all_data_bikes]
filtered_data_weather = [df.drop_duplicates() for df in all_data_weather]

# nettoyage des données
cleaned_data_bikes = [clean_data(df) for df in filtered_data_bikes]
cleaned_data_weather = [clean_data(df) for df in filtered_data_weather]

# Merge bike station data with bike data based on 'id'
merged_data_bikes = []
for df in cleaned_data_bikes:
    if 'id' in bike_stations.columns and df.columns[1] in df.columns:
        merged_df = pd.merge(df, bike_stations, left_on=df.columns[1], right_on='id')
        merged_data_bikes.append(merged_df)
    else:
        print(f"Skipping merge for dataframe due to missing columns: {df.columns[1]} or 'id'")

        # Create a base map
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)  # Centered on France

        # Add city markers with bike availability coloration
        for df in merged_data_bikes:
            for _, row in df.iterrows():
                city = row['city']
                station_id = row['station_id']
                bike_available = row['bike_available']
            
            # Determine marker color based on bike availability
            if bike_available > 10:
                marker_color = 'blue'
            elif bike_available > 5:
                marker_color = 'purple'
            else:
                marker_color = 'red'
            
            # Add marker to the map
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"City: {city}<br>Station ID: {station_id}<br>Bikes Available: {bike_available}",
                icon=folium.Icon(color=marker_color)
            ).add_to(m)

        # Save the map to an HTML file
        m.save('bike_availability_map.html')
        # Create a base map
        m = folium.Map(location=[48.8566, 2.3522], zoom_start=5)  # Centered on France

        # Add city markers with bike availability coloration
        for df in merged_data_bikes:
            for _, row in df.iterrows():
                city = row['city']
                station_id = row['station_id']
                bike_available = row['bike_available']
                
                # Determine marker color based on bike availability
                if bike_available > 10:
                    marker_color = 'blue'
                elif bike_available > 5:
                    marker_color = 'purple'
                else:
                    marker_color = 'red'
                
                # Add marker to the map
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"City: {city}<br>Station ID: {station_id}<br>Bikes Available: {bike_available}",
                    icon=folium.Icon(color=marker_color)
                ).add_to(m)

        # Save the map to an HTML file
        m.save('bike_availability_map.html')

# Afficher les premières lignes de chaque DataFrame dans merged_data_bikes
for df in merged_data_bikes:
    print(df.head())
