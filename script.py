from tools import *
from data import data
import folium
import webbrowser


# Demander à l'utilisateur de spécifier l'heure (exemple: "2022-12-25 14:00")
specified_time = "2022-06-02 19:58"
specified_time = datetime.strptime(specified_time, '%Y-%m-%d %H:%M')
formatted_time = specified_time.strftime('%Y-%m-%dT%H')

# Chemin du répertoire
path = r"C:\Users\weyth\Downloads\Python-20250115"
data_directory = r"C:\Users\weyth\Downloads\Python-20250115\data"
correct_header_data_bikes = ['city', 'id', 'request_date', 'datetime', 'bikes']

data_station = tools.open_all_files_in_directory(path, "bike_station", '\t')
data_weather = tools.open_all_files_in_directory(data_directory, f"data_weather_{formatted_time}", ',')
data_bike = tools.open_all_files_in_directory(data_directory, f"data_bike_{formatted_time}", '\t')
data_pollution = tools.open_all_files_in_directory(data_directory, f"data_pollution_{formatted_time}", ',')

[df.replace_headers_and_add_original(correct_header_data_bikes) for df in data_bike]

[df.filter_dataframes('city', ['amiens', 'marseille']) for df in data_station]
[df.filter_dataframes('city', ['amiens', 'marseille']) for df in data_weather]
[df.filter_dataframes('city', ['amiens', 'marseille']) for df in data_bike]

# Call the function to create a bike station map
[tools.create_bike_station_map(data_station.data, data_bike.data, specified_time) for data_station, data_bike in zip(data_station, data_bike)]

