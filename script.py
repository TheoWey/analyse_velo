import numpy as np
import pandas as pd
import os
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as plt

"""cette fonction ouvre tous les fichiers dans un répertoire spécifié qui 
contiennent un nom spécifique et ont une extension .txt"""
def open_all_files_in_directory(directory_path, name, separator):
    # Création d'un DataFrame vide pour stocker les données
    data = pd.DataFrame()

"""cette fonction ouvre un fichier spécifié et retourne un DataFrame
DataFrame = structure de données bidimensionnelle, genre tableau...etc
mutable et hétérogène, avec des axes étiquetés (lignes et colonnes)."""
def open_file(path, separator):
    try:
        # Lecture du fichier avec gestion des exceptions
        print(f"Reading file: {path}")
        data = pd.read_csv(filepath_or_buffer=path, sep=separator)
    except pd.errors.EmptyDataError:
        # Si le fichier n'a pas de colonnes/ est vide, on retourne un dataFrame vide
        print(f"No columns to parse from file: {path}")
        return pd.DataFrame()
    except Exception as e:
        # Si une autre erreur est detectée pendant la lecture du fichier, on l'affiche et on retourne un DataFrame vide
        print(f"Error reading file {path}: {e}")
        return pd.DataFrame()
    return data

"""Cette fonction ouvre tous les fichiers dans un répertoire spécifié qui 
contiennent un nom spécifique et ont une extension .txt"""
def open_all_files_in_directory(directory_path, name, separator):
    all_data = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt") and name in filename:
            file_path = os.path.join(directory_path, filename)
            data = open_file(file_path, separator)
            all_data.append(data)
    return all_data

 # Remplacer les en-têtes et ajouter les en-têtes originaux comme nouvelle ligne
def replace_headers_and_add_original(all_data, correct_header):
    # Copier l'en-tête actuel de all_data_bikes avant de le remplacer
    current_headers = [df.columns.tolist() for df in all_data]

    # Replace the header of all_data_bikes with the correct one
    all_data = [df.rename(columns=dict(zip(df.columns, correct_header))) for df in all_data]

    # Add the current headers as a new row in each dataframe
    for i, df in enumerate(all_data):
        new_row = pd.DataFrame([current_headers[i]], columns=correct_header)
        all_data[i] = pd.concat([new_row, df], ignore_index=True)
    
    return all_data

""" Là on va filtrer les dataframes selon les valeurs d'une colonne spécifique"""
def filter_dataframes(dataframes, filter_column, filter_values):
    filtered_data_weather = []
    for df in dataframes:
        if filter_column in df.columns:
            # Filtrer les lignes selon les valeurs spécifiées
            filtered_df = df[df[filter_column].isin(filter_values)]
            filtered_data_weather.append(filtered_df)
    return filtered_data_weather

def clean_data(df):
    # enlever les doublons
    df = df.drop_duplicates()
    
    # prendre en charge les valeurs manquantes (exemple: supprimer les lignes avec des valeurs manquantes)
    df = df.dropna()  
    # prendre en charge les valeurs aberrantes 
    if 'temp' in df.columns:
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce') 	
        df = df[(df['temp'] >= -50) & (df['temp'] <= 50)] 
    # si il y a la colonne temp => convertir la colonne "temp" en numérique puis supprimer les valeurs aberrantes et retourner le dataframe
    
    return df 

# Analyse stat univariée
def univariate_analysis(df, column):
    return df[column].describe()

# Analyse stat bivariée
def bivariate_analysis(df, column1, column2):
    return df[[column1, column2]].describe()
# Analyse de corrélation (attention peut bugger)
def correlation_analysis(df1, df2, column1, column2):
    merged_df = pd.merge(df1, df2, on='timestamp')
    correlation = merged_df[[column1, column2]].corr()
    return correlation

# definir une fonction pour analyser les données de vélos
def analyze_bike_data(df):
    # regrouper les données par heure pour trouver les heures de pointe
    df['hour'] = df['timestamp'].dt.hour
    peak_hours = df.groupby('hour').size()
    
    # regrouper les données par jour de la semaine pour trouver les jours de la semaine les plus utilisés
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    weekdays_vs_weekends = df.groupby('day_of_week').size()
    
    return peak_hours, weekdays_vs_weekends

# definir une fonction pour projeter l'utilisation des vélos
def project_bike_usage(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    
    model = LinearRegression()
    model.fit(X, y)
    # tracé du modèle de regression linéaire     
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

# remplacer les en-têtes et ajouter les en-têtes originaux comme nouvelle ligne
all_data_bikes = replace_headers_and_add_original(all_data_bikes, correct_header_data_bikes)

# suppression des doublons
bike_stations = bike_stations.drop_duplicates()
filtered_data_bikes = [df.drop_duplicates() for df in all_data_bikes]
filtered_data_weather = [df.drop_duplicates() for df in all_data_weather]

# nettoyage des données
cleaned_data_bikes = [clean_data(df) for df in filtered_data_bikes]
cleaned_data_weather = [clean_data(df) for df in filtered_data_weather]

# fusionner les données de vélos avec les données des stations de vélos
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


# Vérifier si merged_data_bikes est une liste
if isinstance(merged_data_bikes, list):
    # Concaténer la liste si elle contient des DataFrames
    merged_data_bikes = pd.concat(merged_data_bikes, ignore_index=True)

# Vérifier que merged_data_bikes est maintenant un DataFrame
if not isinstance(merged_data_bikes, pd.DataFrame):
    raise ValueError("merged_data_bikes doit être un DataFrame après concaténation.")

# Vérifier si la colonne 'timestamp' existe
if 'request_date' not in merged_data_bikes.columns:
    raise KeyError("'request_date' column is missing from merged_data_bikes")

# Convertir les dates en format datetime si ce n'est pas déjà fait
merged_data_bikes['request_date'] = pd.to_datetime(merged_data_bikes['request_date'], errors='coerce')

# Supprimer les lignes avec des valeurs NaT dans 'request_date'
merged_data_bikes = merged_data_bikes.dropna(subset=['request_date'])

# Filtrer les données pour obtenir celles à 9h
merged_data_bikes['hour'] = merged_data_bikes['request_date'].dt.hour
merged_data_bikes['month'] = merged_data_bikes['request_date'].dt.month
filtered_data = merged_data_bikes[merged_data_bikes['hour'] == 9]

# Vérifier les données filtrées
print("Filtered Data:")
print(filtered_data['city', 'bike_available', 'hour', 'month'].head())

# Calcul des statistiques de base pour chaque mois
confidence_intervals = {}
for month, group in filtered_data.groupby('month'):
    print(f"Processing month: {month}")
    mean_bikes = group['bike_available'].mean()
    std_bikes = group['bike_available'].std()
    n = len(group)

    # Calcul de l'intervalle de confiance à 95%
    alpha = 0.05
    t_value = t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0
    marge_erreur = t_value * (std_bikes / np.sqrt(n)) if n > 1 else 0

    confidence_interval = (mean_bikes - marge_erreur, mean_bikes + marge_erreur)
    confidence_intervals[month] = {
        'mean': mean_bikes,
        'std': std_bikes,
        'n': n,
        'interval': confidence_interval
    }


# Préparer les données pour la representation graphique
months = list(confidence_intervals.keys())
means = [stats['mean'] for stats in confidence_intervals.values()]
erreur = [(stats['interval'][1] - stats['mean']) for stats in confidence_intervals.values()]

# Vérifier les données pour le graphique
print("Months:", months)
print("Means:", means)
print("Errors:", erreur)
# Afficher les résultats
for month, stats in confidence_intervals.items():
    print(f"Mois {month}:")
    print(f"  Moyenne: {stats['mean']:.2f}")
    print(f"  Écart-Type: {stats['std']:.2f}")
    print(f"  Taille de l'échantillon: {stats['n']}")
    print(f"  Intervalle de Confiance: {stats['interval']}")
    print("-")

# Créer le graphique
plt.figure(figsize=(10, 6))
plt.bar(months, means, yerr=errors, capsize=5, color='skyblue', alpha=0.7)
plt.xlabel('Mois')
plt.ylabel('Nombre moyen de vélos disponibles')
plt.title('Nombre moyen de vélos disponibles à 9h avec intervalle de confiance à 95%')
plt.xticks(months)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Afficher le graphique
plt.show()
"""