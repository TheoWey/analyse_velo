import numpy as np
import pandas as pd
import os
import mpld3
from mpld3 import plugins
from datetime import datetime
import seaborn as sns  # pour la heatmap et les boxplots
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
# Fusionner les DataFrames de données vélo en un seul DataFrame
bike_df = open_file(r"C:\Users\ingri\Documents\analyse_velo\merged_data_bikes.txt", '\t')
print(bike_df.head())
# Conversion de la colonne 'request_date' en datetime
bike_df['request_date'] = pd.to_datetime(bike_df['request_date'])

# Création de colonnes supplémentaires pour l'analyse
bike_df['year'] = bike_df['request_date'].dt.year
bike_df['month'] = bike_df['request_date'].dt.month
bike_df['day'] = bike_df['request_date'].dt.day
bike_df['hour'] = bike_df['request_date'].dt.hour
bike_df['day_of_week'] = bike_df['request_date'].dt.dayofweek  # 0 = lundi, 6 = dimanche
bike_df['time'] = bike_df['request_date'].dt.time

timeday= bike_df['request_date']
year= bike_df['year']
month= bike_df['month']
day= bike_df['day']
hour= bike_df['hour']
weekday= bike_df['day_of_week']
time= bike_df['time']

def get_season(month):
    if month in [12, 1, 2]:
        return "Hiver"
    elif month in [3, 4, 5]:
        return "Printemps"
    elif month in [6, 7, 8]:
        return "Été"
    else:
        return "Automne"

bike_df['season'] = bike_df['month'].apply(get_season)
bike_df['periode_jour_nuit'] = bike_df['hour'].apply(lambda h: "Jour" if 6 <= h < 18 else "Nuit")
bike_df['weekday_weekend'] = bike_df['day_of_week'].apply(lambda d: "Weekend" if d >= 5 else "Weekday")
bike_df['peak_offpeak'] = bike_df['hour'].apply(lambda h: "Heures de Pointe" if (7 <= h < 9 or 17 <= h < 19) else "Heures Creuses")

season= bike_df['season']
daynight= bike_df['periode_jour_nuit']
weekday_weekend= bike_df['weekday_weekend']
peak_offpeak= bike_df['peak_offpeak']

city_x = bike_df["city_x"]
station_id = bike_df["station_id"]
request_date = bike_df["request_date"]
answer_date = bike_df["answer_date"]
bike_available = bike_df["bike_available"]
name = bike_df["name"]
address = bike_df["address"]
banking = bike_df["banking"]
bonus = bike_df["bonus"]
bike_stands = bike_df["bike_stands"]
available_bike_stands = bike_df["available_bike_stands"]
available_bikes = bike_df["available_bikes"]
status = bike_df["status"]
last_update = bike_df["last_update"]
city_y = bike_df["city_y"]
id_ = bike_df["id"]
company = bike_df["company"]
latitude = bike_df["latitude"]
longitude = bike_df["longitude"]
altitude = bike_df["altitude"]
id_pollution = bike_df["id_pollution"]
dist_bike_pollution = bike_df["dist_bike_pollution"]

print(longitude)
def save_fig_jpg(fig, filename):
    fig.savefig(filename, format='jpg')

# Style des graphes
sns.set_style("whitegrid")

# Utilisation des vélos par mois
plt.figure(figsize=(12,6))
for month in range(1, 13):
    monthly_data = bike_df[bike_df['month'] == month]
    plt.plot(monthly_data['hour'], monthly_data['bike_available'], label=f'Mois {month}')
plt.xlabel("Heure")
plt.ylabel("Nombre de vélos disponibles")
plt.title("Évolution de l'utilisation des vélos par mois")
plt.legend(title="Mois")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("utilisation_velos_par_mois.jpg")

# Distribution Jour vs Nuit
plt.figure(figsize=(8,6))
sns.countplot(x=daynight, order=["Jour", "Nuit"], palette=["skyblue", "navy"])
plt.title("Distribution des vélos disponibles - Jour vs Nuit")
plt.xlabel("Période")
plt.ylabel("Nombre d'enregistrements")
plt.tight_layout()
save_fig_jpg(plt.gcf(), "jour_nuit.jpg")

# Distribution Semaine vs Week-end
plt.figure(figsize=(8,6))
sns.countplot(x=weekday_weekend, order=["Weekday", "Weekend"], palette=["green", "red"])
plt.title("Distribution des vélos disponibles - Semaine vs Week-end")
plt.xlabel("Période")
plt.ylabel("Nombre d'enregistrements")
plt.tight_layout()
save_fig_jpg(plt.gcf(), "semaine_weekend.jpg")

# Heures de Pointe vs Heures Creuses
plt.figure(figsize=(8,6))
sns.countplot(x=peak_offpeak, order=["Heures de Pointe", "Heures Creuses"], palette=["orange", "gray"])
plt.title("Disponibilité des vélos - Heures de Pointe vs Heures Creuses")
plt.xlabel("Période")
plt.ylabel("Nombre d'enregistrements")
plt.tight_layout()
save_fig_jpg(plt.gcf(), "heures_pointe.jpg")

# Répartition par Saison
plt.figure(figsize=(8,6))
order = ["Hiver", "Printemps", "Été", "Automne"]
sns.countplot(x=season, order=order, palette=["blue", "green", "orange", "brown"])
plt.title("Répartition des vélos disponibles selon la saison")
plt.xlabel("Saison")
plt.ylabel("Nombre d'enregistrements")
plt.tight_layout()
save_fig_jpg(plt.gcf(), "saison.jpg")

# Utilisation moyenne par Station
plt.figure(figsize=(12,6))
station_usage = bike_df.groupby("station_id")["bike_available"].mean().reset_index()
sns.barplot(x="station_id", y="bike_available", data=station_usage, palette="viridis")
plt.title("Nombre moyen de vélos disponibles par station")
plt.xlabel("Station")
plt.ylabel("Nombre moyen de vélos")
plt.xticks(rotation=90)
plt.tight_layout()
save_fig_jpg(plt.gcf(), "station.jpg")

# Comparaison de l'utilisation entre les Villes
plt.figure(figsize=(8,6))
city_usage = bike_df.groupby("city_x")["bike_available"].mean().reset_index()
sns.barplot(x="city_x", y="bike_available", data=city_usage, palette="coolwarm")
plt.title("Comparaison de l'utilisation des vélos entre villes")
plt.xlabel("Ville")
plt.ylabel("Nombre moyen de vélos")
plt.tight_layout()
save_fig_jpg(plt.gcf(), "ville.jpg")

# Analyse des stations avec le plus de vélos disponibles
plt.figure(figsize=(12,6))
top_stations = station_usage.sort_values(by="bike_available", ascending=False).head(10)
sns.barplot(x="station_id", y="bike_available", data=top_stations, palette="magma")
plt.title("Top 10 des stations avec le plus de vélos disponibles")
plt.xlabel("Station")
plt.ylabel("Nombre moyen de vélos")
plt.xticks(rotation=90)
plt.tight_layout()
save_fig_jpg(plt.gcf(), "top_stations.jpg")
