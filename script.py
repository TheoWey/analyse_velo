import numpy as np
import pandas as pd
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap

# Demander à l'utilisateur de spécifier l'heure (exemple: "2022-12-25 14:00")
specified_time = "2022-06-02 19:58"
specified_time = datetime.strptime(specified_time, '%Y-%m-%d %H:%M')
formatted_time = specified_time.strftime('%Y-%m-%dT%H')
print(formatted_time)


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


#Ouvre un fichier spécifié ou tout fichier contenant le date de discrimination
def open_all_files_in_directory(directory_path, name, separator):
    print(name)
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") and name in filename:
            file_path = os.path.join(directory_path, filename)
            data = open_file(file_path, separator)

                
            all_data.append(data)
               
    return all_data


#fonction pour filtrer les donnée pour garder uniquement marseille et amiens
def filter_dataframes(dataframes, filter_column, filter_values):
    filtered_data = []
    for df in dataframes:
        if filter_column in df.columns:
            # Filtrer les lignes selon les valeurs spécifiées
            filtered_df = df[df[filter_column].isin(filter_values)]
            filtered_data.append(filtered_df)
    return filtered_data

# Fonction pour obtenir les données de vélos à un moment donné
def get_bike_count_at_time(bike_data_list, station_id, time):
    """
    bike_data_list : liste de DataFrames
    station_id : ID de la station à chercher
    time : heure spécifique pour obtenir les données
    """
    for bike_data in bike_data_list:  # Boucle sur chaque DataFrame    
        # Convertir la colonne datetime en format datetime pour comparaison
        bike_data.loc[:, 'datetime'] = pd.to_datetime(bike_data['datetime'], errors='coerce')  # Utilise 'coerce' pour gérer les erreurs de conversion
        
        # Filtrer les données pour la station et l'heure spécifiée
        station_data = bike_data[(bike_data['id'] == station_id) & (bike_data['datetime'] == time)]
        
        if not station_data.empty:
            return station_data.iloc[0]['bikes']  # Retourne le nombre de vélos trouvés

    return None  # Si aucune donnée n'est trouvée dans tous les DataFrames



# Chemin du répertoire
data_directory = r"C:\Users\carra\Downloads\Python-20250115\data"
data_station_directory =r"C:\Users\carra\Downloads\Python-20250115"

# Charger les données \n for tab
all_data_weather = open_all_files_in_directory(data_directory, f"data_weather_{formatted_time}", ',')
all_data_bike = open_all_files_in_directory(data_directory, f"data_bike_{formatted_time}", '\t')
all_data_station = open_all_files_in_directory(data_station_directory, "bike_station.txt", '\t')

#Copy the current header of all_data_bikes before replacing it
current_headers = [df.columns.tolist() for df in all_data_bike]

#Replace the header of all_data_bikes with the correct one
correct_header = ['name', 'id', 'request_date', 'datetime', 'bikes']
all_data_bike = [df.rename(columns=dict(zip(df.columns, correct_header))) for df in all_data_bike]

#Add the current headers as a new row in each dataframe
for i, df in enumerate(all_data_bike):
    new_row = pd.DataFrame([current_headers[i]], columns=correct_header)
    all_data_bike[i] = pd.concat([new_row, df], ignore_index=True)


print(all_data_bike)
# Filtrer les données pour garder uniquement "amiens" et "marseille"
filtered_data_weather = filter_dataframes(all_data_weather, filter_column='name', filter_values=['Amiens','Marseille'])
filtered_data_bike = filter_dataframes(all_data_bike, filter_column='name', filter_values=['amiens','marseille'])
filtered_data_station = filter_dataframes(all_data_station, filter_column='city', filter_values=['amiens','marseille'])

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
print("filtered data bike")
print(filtered_data_bike)
# Add markers with color 
for lat, lon, station_id, places in normalized_data :
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

# Afficher les résultats
print(filtered_data_station)
