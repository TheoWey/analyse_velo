import numpy as np
import pandas as pd
import os
import folium
from folium.plugins import HeatMap

def open_file(path, separator):
    try:
        # Lecture du fichier avec gestion des exceptions
        data = pd.read_csv(filepath_or_buffer=path, sep=separator)
    except pd.errors.EmptyDataError:
        print(f"No columns to parse from file: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        return pd.DataFrame()
    
    # Remplacer les points par des virgules dans les chaînes de caractères
    data = data.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, str) else x)
    return data

def open_all_files_in_directory(directory_path, name, separator):
    all_data_weather = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") and name in filename:
            file_path = os.path.join(directory_path, filename)
            data = open_file(file_path, separator)
            all_data_weather.append(data)
    return all_data_weather

def filter_dataframes(dataframes, filter_column, filter_values):
    filtered_data_weather = []
    for df in dataframes:
        if filter_column in df.columns:
            # Filtrer les lignes selon les valeurs spécifiées
            filtered_df = df[df[filter_column].isin(filter_values)]
            filtered_data_weather.append(filtered_df)
    return filtered_data_weather

# Chemin du répertoire
data_directory = r"C:\Users\weyth\Downloads\Python-20250110\data"

# Charger les données \n for tab
all_data_weather = open_all_files_in_directory(data_directory, "data_weather_2022-12-25", ',')
all_data_bike = open_all_files_in_directory(data_directory, "data_bike_2022-12-25", '\t')
# Filtrer les données pour garder uniquement "amiens" et "marseille"
filtered_data_weather = filter_dataframes(all_data_weather, filter_column='name', filter_values=['Amiens','Marseille'])
filtered_data_bike = filter_dataframes(all_data_bike, filter_column='name', filter_values=['Amiens','Marseille'])

# Extract lon and lat columns
locations = []
for df in filtered_data_weather:
    if 'lon' in df.columns and 'lat' in df.columns:
        for _, row in df.iterrows():
            try:
                lat = float(row['lat'].replace(',', '.'))
                lon = float(row['lon'].replace(',', '.'))
                locations.append((lat, lon))
            except ValueError:
                print(f"Invalid data for lat/lon: {row['lat']}, {row['lon']}")

# Create a map centered around the average location
if locations:
    avg_lat = np.mean([loc[0] for loc in locations])
    avg_lon = np.mean([loc[1] for loc in locations])
    map_center = [avg_lat, avg_lon]
else:
    map_center = [0, 0]
print("Average location:", map_center)
map = folium.Map(location=map_center, zoom_start=6)

# Add points to the map
for lat, lon in locations:
    folium.Marker(location=[lat, lon]).add_to(map)
    # Create a heatmap layer

# Save the map to an HTML file
map.save('map.html')


# Afficher les résultats
print(filtered_data_weather)
