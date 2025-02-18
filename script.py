import numpy as np
import pandas as pd
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt


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
                    <div style="width: {percentage}%; background-color:rgb(18, 38, 36); height: 20px; border-radius: 5px;"></div>
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


def correlation_analysis(df1, df2, column1, column2):
    merged_df = pd.merge(df1, df2, on='timestamp')
    correlation = merged_df[[column1, column2]].corr()
    return correlation

def extract_monthly_weather_files_using_existing_functions(data_directory, month, year, city, hour=12):
    """
    Extrait et combine tous les fichiers météo pour un mois donné,
    sélectionne une seule ligne par jour à une heure donnée, 
    et filtre par ville spécifiée (Amiens ou Marseille).

    Args:
        data_directory (str): Chemin vers le répertoire des données météo.
        month (str): Mois à extraire (format "MM").
        year (str): Année à extraire (format "YYYY").
        city (str): Ville à filtrer ('Amiens' ou 'Marseille').
        hour (int): Heure à fixer pour l'extraction (par défaut 12h).

    Returns:
        pd.DataFrame: DataFrame combiné avec une seule ligne par jour pour la ville et l'heure spécifiées.
    """
    formatted_month = f"{year}-{month}"
    
    # Extraire tous les fichiers météo pour le mois
    all_data_weather = open_all_files_in_directory(data_directory, f"data_weather_{formatted_month}", ',')
    
    # Filtrer les données pour la ville spécifiée
    filtered_weather_data = filter_dataframes(all_data_weather, filter_column='name', filter_values=[city])
    
    if not filtered_weather_data:
        print(f"No weather data found for {city} in {formatted_month}.")
        return pd.DataFrame()

    # Combiner les données filtrées en un seul DataFrame
    combined_weather_data = pd.concat(filtered_weather_data, ignore_index=True)
    
    # Vérifier si les colonnes nécessaires sont présentes
    if 'request_time' not in combined_weather_data.columns:
        print("Missing 'request_time' column in the weather data.")
        return pd.DataFrame()
    
    # Convertir le timestamp en datetime pour faciliter le filtrage
    combined_weather_data['request_time'] = pd.to_datetime(combined_weather_data['request_time'], errors='coerce')
    
    # Ajouter une colonne "date" pour regrouper par jour
    combined_weather_data['date'] = combined_weather_data['request_time'].dt.date
    
    # Ajouter une colonne "hour" pour filtrer l'heure spécifique
    combined_weather_data['hour'] = combined_weather_data['request_time'].dt.hour
    
    # Filtrer pour l'heure spécifiée
    daily_weather_at_hour = combined_weather_data[combined_weather_data['hour'] == hour]
    
    # Garder une seule ligne par jour
    final_weather_data = daily_weather_at_hour.groupby('date').first().reset_index()
    
    # Supprimer les colonnes inutiles si nécessaire
    final_weather_data = final_weather_data.drop(columns=['hour'], errors='ignore')
    
    return final_weather_data


def calculate_daily_bike_total_at_12h(data_directory, month, year, ville, nbr_velo):
    year = int(year)
    month = int(month)

    formatted_month = f"{year}-{month:02d}"
    all_data_bike = open_all_files_in_directory(data_directory, f"data_bike_{formatted_month}", '\t')
    all_data_bike = replace_headers_and_add_original(all_data_bike, correct_header_data_bikes)

    filtered_data_bike = filter_dataframes(all_data_bike, filter_column='city', filter_values=[ville])

    daily_totals = []
    
    

    for day in range(1, 32):
        try:
            current_date = datetime(year, month, day).date()
            target_time = datetime(year, month, day, 12, 0, 0)

            day_files = []
            for df in filtered_data_bike:
                if 'datetime' in df.columns:
                    # Crée une copie explicite du DataFrame
                    df = df.copy()
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    day_files.append(df[df['datetime'] == target_time])

            day_files = [df for df in day_files if not df.empty]
            if day_files:
                daily_data = pd.concat(day_files, ignore_index=True)

                # Vérifier si la colonne 'bikes' existe et la convertir en numérique
                if 'bikes' in daily_data.columns:
                    daily_data['bikes'] = pd.to_numeric(daily_data['bikes'], errors='coerce')

                    # Calculer le total des vélos disponibles à 12h
                    total_bikes_at_12h = daily_data['bikes'].sum()
                    daily_totals.append({'city': ville, 'date': current_date, 'total_bikes': nbr_velo-total_bikes_at_12h})
                else:
                    # Si la colonne 'bikes' n'existe pas, l'ajouter avec une valeur de 0
                    daily_totals.append({'city': ville, 'date': current_date, 'total_bikes': nbr_velo})

        except ValueError:
            continue

    if daily_totals:
        combined_daily_totals = pd.DataFrame(daily_totals)
    else:
        combined_daily_totals = pd.DataFrame(columns=['city', 'date', 'total_bikes'])

    return combined_daily_totals






# Demander à l'utilisateur de spécifier l'heure (exemple: "2022-12-25 14:00:00")
specified_time = "2022-10-09 19:58"
specified_time = datetime.strptime(specified_time, '%Y-%m-%d %H:%M')
formatted_time = specified_time.strftime('%Y-%m-%dT%H')
formatted_weather_pollution = specified_time.strftime('%Y-%m-%d')

years = specified_time.strftime('%Y')
month = specified_time.strftime('%m')
# Chemin du répertoire
path = r"C:\Users\carra\Downloads\Python-20250115"
data_directory = r"C:\Users\carra\Downloads\Python-20250115\data"
correct_header_data_bikes = ['city', 'id', 'request_date', 'datetime', 'bikes']

# lecture des fichiers de données
all_data_station = open_all_files_in_directory(path, "bike_station.txt", '\t')
all_data_weather = open_all_files_in_directory(data_directory, f"data_weather_{formatted_weather_pollution}", ',')
all_data_bike = open_all_files_in_directory(data_directory, f"data_bike_{formatted_time}", '\t')


# Replace headers and add original headers as a new row
all_data_bike = replace_headers_and_add_original(all_data_bike, correct_header_data_bikes)

# selection par ville
filtered_data_station = filter_dataframes(all_data_station, filter_column='city', filter_values=['amiens','marseille'])
filtered_data_bike = filter_dataframes(all_data_bike, filter_column='city', filter_values=['amiens','marseille'])
filtered_data_weather = filter_dataframes(all_data_weather, filter_column='name', filter_values=['Amiens','Marseille'])

filtered_data_station_amiens = filter_dataframes(filtered_data_station, filter_column='city', filter_values=['amiens'])

#suppression des doublons
filtered_data_station = [df.drop_duplicates() for df in filtered_data_station]
filtered_data_bike = [df.drop_duplicates() for df in filtered_data_bike]
filtered_data_weather = [df.drop_duplicates() for df in filtered_data_weather]

print(filtered_data_station_amiens)
# Conversion en numérique de la colonne 'bike_stands', en remplaçant les erreurs par NaN


for df in filtered_data_station_amiens:
   print(df['bike_stands'])
   
nbr_velo_amiens = filtered_data_station_amiens[0]['bike_stands'].sum()

# Affichage du résultat
print(nbr_velo_amiens)
# Call the function
#create_bike_station_map(filtered_data_station, filtered_data_bike, specified_time)

meteo = extract_monthly_weather_files_using_existing_functions(data_directory, month, years, 'Amiens')


velo = (calculate_daily_bike_total_at_12h(data_directory, month, years, 'amiens', nbr_velo_amiens))


# Extraire la colonne 'total_bikes' de 'velo'
total_bikes = velo['total_bikes']

# Reshaper la colonne pour s'assurer que l'index est bien aligné
total_bikes = total_bikes.reset_index(drop=True)

# Concatenation des deux DataFrames
meteo = pd.concat([meteo, total_bikes], axis=1)

# Vérifier le résultat
print(meteo)

meteo = meteo[['temp','feels_like','temp_min','temp_max','pressure','humidity','sea_level','grnd_level','speed','deg','gust','clouds','total_bikes']]
meteo = meteo.apply(pd.to_numeric, errors='coerce')
df_numerique = meteo.select_dtypes(include=['float64', 'int64'])
# Calculer la matrice de corrélation
corr_matrix = df_numerique.corr()

# Afficher la matrice de corrélation avec seaborn
plt.figure(figsize=(10, 8))  # Taille de la figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()
