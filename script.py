import numpy as np
import pandas as pd
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    # Copy the current header of all_data_bike before replacing it
    current_headers = [df.columns.tolist() for df in all_data]

    # Replace the header of all_data_bike with the correct one
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

def merge_bike_and_station_data(data_1, data_2, merge_key):
    merged_data = []
    for df in data_1:
        for station_df in data_2:
            if merge_key in station_df.columns and df.columns[1] in df.columns:
                merged_df = pd.merge(df, station_df, left_on=df.columns[1], right_on=merge_key)
                merged_data.append(merged_df)
            else:
                print(f"Skipping merge for dataframe due to missing columns: {df.columns[1]} or '{merge_key}'")
    return merged_data

# Fonction pour obtenir les données de vélos à un moment donné
def get_bike_count_at_time(bike_data_list, station_id, time):
    """
    bike_data_list : liste de DataFrames
    station_id : ID de la station à chercher
    time : heure spécifique pour obtenir les données
    """
    for bike_data in bike_data_list:  # Boucle sur chaque DataFrame    
        # Convertir la colonne datetime en format datetime pour comparaison
        if 'datetime' in bike_data.columns:
            bike_data.loc[:, 'datetime'] = pd.to_datetime(bike_data['datetime'], errors='coerce')  # Utilise 'coerce' pour gérer les erreurs de conversion
        else:
            continue
        
        # Filtrer les données pour la station et l'heure spécifiée
        station_data = bike_data[(bike_data['id'] == station_id) & (bike_data['datetime'] == time)]
        
        if not station_data.empty:
            return station_data.iloc[0]['bikes']  # Retourne le nombre de vélos trouvés

    return None  # Si aucune donnée n'est trouvée dans tous les DataFrames

def create_bike_station_map(filtered_data_station, filtered_data_bike, specified_time):
    normalized_data = []
    for df in filtered_data_station:
        if all(col in df.columns for col in ['longitude', 'latitude']):
            df = df.dropna(subset=['longitude', 'latitude'])
            for _, row in df.iterrows():
                try:
                    lat = float(str(row['latitude']).replace(',', '.'))
                    lon = float(str(row['longitude']).replace(',', '.'))
                    station_id = row['id']  # ID de la station
                    places = row['bike_stands']
                    normalized_data.append([lat, lon, station_id, places])  # Ajouter l'ID à la liste
                except (ValueError, TypeError) as e:
                    continue      

    map_center = [46.603354, 1.888334]

    # Create map
    map = folium.Map(location=map_center, zoom_start=6)
    # Add markers with color 
    for lat, lon, station_id, places in normalized_data:
        # Appeler la fonction mise à jour pour récupérer le nombre de vélos
        bike_count = get_bike_count_at_time(filtered_data_bike, station_id, specified_time)

        # Ajouter un marqueur pour chaque station
        if bike_count is not None:
            # Calculer le pourcentage de vélos disponibles
            percentage = (bike_count / int(places)) * 100

            # Créer le code HTML pour le popup avec une barre de progression
            popup_html = f"""
            <div style="width: 200px;">
                <h4>Station ID: {station_id}</h4>
                <p>Vélos disponibles: {bike_count}/{int(places)}</p>
                <div style="background-color: #e0e0e0; border-radius: 5px; padding: 2px;">
                    <div style="width: {percentage}%; background-color: #76c7c0; height: 20px; border-radius: 5px;"></div>
                </div>
            </div>
            """
            color = "green" if bike_count > 5 else "orange" if bike_count > 0 else "red"
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
            icon=folium.Icon(icon="bicycle", prefix="fa", color=color)
        ).add_to(map)

    # Save the map to an HTML file
    map.save('velo_map.html')

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
specified_time = "2022-06-02 19:58"
specified_time = datetime.strptime(specified_time, '%Y-%m-%d %H:%M')
formatted_time = specified_time.strftime('%Y-%m-%dT%H')

# Chemin du répertoire
path = r"C:\Users\ingri\Downloads\data"
data_directory = r"C:\Users\ingri\Downloads\data"
correct_header_data_bikes = ['city', 'id', 'request_date', 'datetime', 'bikes']

# lecture des fichiers de données
all_data_station = open_all_files_in_directory(path, "bike_station.txt", '\t')
all_data_weather = open_all_files_in_directory(data_directory, f"data_weather_{formatted_time}", ',')
all_data_bike = open_all_files_in_directory(data_directory, f"data_bike_{formatted_time}", '\t')

# Replace headers and add original headers as a new row
all_data_bike = replace_headers_and_add_original(all_data_bike, correct_header_data_bikes)

# selection par ville
filtered_data_station = filter_dataframes(all_data_station, filter_column='city', filter_values=['amiens','marseille'])
filtered_data_bike = filter_dataframes(all_data_bike, filter_column='city', filter_values=['amiens','marseille'])
filtered_data_weather = [df.drop_duplicates() for df in all_data_weather]

#suppression des doublons
filtered_data_station = [df.drop_duplicates() for df in filtered_data_station]
filtered_data_bike = [df.drop_duplicates() for df in filtered_data_bike]
filtered_data_weather = [df.drop_duplicates() for df in filtered_data_weather]

# nettoyage des données
cleaned_data_bikes = [clean_data(df) for df in filtered_data_bike]
cleaned_data_weather = [clean_data(df) for df in filtered_data_weather]
# Créer la carte des stations de vélos
create_bike_station_map(filtered_data_station, filtered_data_bike, specified_time)

#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

if filtered_data_weather:
    weather_df = pd.concat(filtered_data_weather, ignore_index=True)
else:
    print("Aucun DataFrame à concaténer dans filtered_data_weather.")

# Analyse des données météorologiques
weather_df = pd.concat(filtered_data_weather, ignore_index=True)

# Vérifier si la colonne 'humidity' existe, sinon la créer avec des valeurs NaN
if 'humidity' not in weather_df.columns:
    weather_df['humidity'] = np.nan

# Analyse univariée
univariate_temp = univariate_analysis(weather_df, 'temp')
univariate_humidity = univariate_analysis(weather_df, 'humidity')

# Analyse bivariée
bivariate_temp_humidity = bivariate_analysis(weather_df, 'temp', 'humidity')

# Générer des graphiques
plt.figure(figsize=(10, 6))
plt.hist(weather_df['temp'].dropna(), bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution de la température')
plt.xlabel('Température (°C)')
plt.ylabel('Fréquence')
plt.savefig('temp_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(weather_df['temp'], weather_df['humidity'], alpha=0.5)
plt.title('Température vs Humidité')
plt.xlabel('Température (°C)')
plt.ylabel('Humidité (%)')
plt.savefig('temp_vs_humidity.png')
plt.close()
# Analyse des données de pollution
pollution_files = open_all_files_in_directory(data_directory, "data_pollution_", ',')
pollution_df_list = [clean_data(df) for df in pollution_files]

# Analyse univariée des données de pollution
univariate_pollution = {}
for df in pollution_df_list:
    for col in df.columns:
        if col not in ['id', 'date']:
            if col not in univariate_pollution:
                univariate_pollution[col] = []
            univariate_pollution[col].append(univariate_analysis(df, col))

# Analyse bivariée des données de pollution (exemple: NO2 vs PM10)
bivariate_pollution = []
for df in pollution_df_list:
    if 'NO2' in df.columns and 'PM10' in df.columns:
        bivariate_pollution.append(bivariate_analysis(df, 'NO2', 'PM10'))

# Générer des graphiques pour les données de pollution
for i, df in enumerate(pollution_df_list):
    plt.figure(figsize=(10, 6))
    plt.hist(df['NO2'].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution de NO2 - Fichier {i+1}')
    plt.xlabel('NO2 (µg/m³)')
    plt.ylabel('Fréquence')
    plt.savefig(f'no2_distribution_{i+1}.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(df['NO2'], df['PM10'], alpha=0.5)
    plt.title(f'NO2 vs PM10 - Fichier {i+1}')
    plt.xlabel('NO2 (µg/m³)')
    plt.ylabel('PM10 (µg/m³)')
    plt.savefig(f'no2_vs_pm10_{i+1}.png')
    plt.close()
