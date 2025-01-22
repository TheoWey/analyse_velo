import numpy as np
import pandas as pd
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap
from sklearn.linear_model import LinearRegression
from data import data

class tools:
    def __init__(self):
        pass

    def open_all_files_in_directory(directory_path, name, separator):
        all_data = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".txt") and name in filename:
                file_path = os.path.join(directory_path, filename)
                data_content = data(file_path, separator)
                all_data.append(data_content)
        return all_data

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
            bike_count = tools.get_bike_count_at_time(filtered_data_bike, station_id, specified_time)

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

    def linear_regression(self, df, x_column, y_column):
        # Linear regression
        X = df[x_column].values.reshape(-1, 1)
        y = df[y_column].values
        model = LinearRegression()
        model.fit(X, y)
        return model
    
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
