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

# Extract temperature and location data with validation
temp_data = []
for df in filtered_data_weather:
    if all(col in df.columns for col in ['lon', 'lat', 'temp']):
        df = df.dropna(subset=['lon', 'lat', 'temp'])
        
        for _, row in df.iterrows():
            try:
                lat = float(str(row['lat']).replace(',', '.'))
                lon = float(str(row['lon']).replace(',', '.'))
                temp = float(str(row['temp']).replace(',', '.'))
                
                if not (np.isnan(lat) or np.isnan(lon) or np.isnan(temp)):
                    temp_data.append([lat, lon, temp])
            except (ValueError, TypeError) as e:
                continue

# Calculate map center and normalize temperatures
if temp_data:
    temp_array = np.array(temp_data)
    min_temp = np.min(temp_array[:, 2])
    max_temp = np.max(temp_array[:, 2])
    
    # Normalize temperatures between 0 and 1
    normalized_data = [[point[0], point[1], (point[2] - min_temp)/(max_temp - min_temp)] 
                      for point in temp_data]
    
    map_center = [np.mean(temp_array[:, 0]), np.mean(temp_array[:, 1])]
else:
    map_center = [46.603354, 1.888334]
    normalized_data = []

# Create map
map = folium.Map(location=map_center, zoom_start=6)

# Add markers with color gradient based on temperature
for lat, lon, temp in normalized_data:
    # Create RGB color (red for high, green for low)
    color = f'#{int(255 * ( temp)):02x}{int(255 * (1 - temp)):02x}00'
    
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        popup=f'Temperature: {temp * (max_temp - min_temp) + min_temp:.2f}°C',
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7
    ).add_to(map)

# Add white markers if no temperature data is present
for df in filtered_data_weather:
    if all(col in df.columns for col in ['lon', 'lat']) and 'temp' not in df.columns:
        for _, row in df.iterrows():
            try:
                lat = float(str(row['lat']).replace(',', '.'))
                lon = float(str(row['lon']).replace(',', '.'))
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=10,
                    popup='No temperature data',
                    color='white',
                    fill=True,
                    fillColor='white',
                    fillOpacity=0.6
                ).add_to(map)
            except (ValueError, TypeError) as e:
                continue

# Save the map to an HTML file
map.save('temperature_gradient_map.html')

# Afficher les résultats
print(filtered_data_weather)
