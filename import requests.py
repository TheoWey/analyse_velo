import requests
import csv
import time
import os
from datetime import datetime
import pandas as pd

def get_api_data(url, api_key):
    headers = {"Accept": "application/json"}
    params = {"apiKey": api_key}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête : {e}")
        return None

def get_station_names(api_key, contract_name):
    url = "https://api.jcdecaux.com/vls/v3/stations"
    headers = {"Accept": "application/json"}
    params = {"apiKey": api_key, "contract": contract_name}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        stations = response.json()
        station_data = {station["number"]: station["name"] for station in stations}
        print(f"Stations pour {contract_name}: {station_data}")
        return station_data
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête pour {contract_name}: {e}")
        return None

def get_station_details(api_key, contract_name, station_number):
    url = f"https://api.jcdecaux.com/vls/v3/stations/{station_number}"
    headers = {"Accept": "application/json"}
    params = {"apiKey": api_key, "contract": contract_name}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        station_info = response.json()
        return station_info
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors de la requête pour la station {station_number} de {contract_name}: {e}")
        return None

def save_station_data_to_xlsx(station_data, contract_name, api_key, date_str):
    # Nom du fichier
    filename = f"databike_{contract_name}_{date_str}.xlsx"
    
    # Créer le dossier "real_data" s'il n'existe pas
    os.makedirs("real_data", exist_ok=True)
    
    # Chemin complet du fichier
    filepath = os.path.join("real_data", filename)
    
    # Liste des colonnes attendues
    headers = [
        "station_number", "station_name", "available_bikes", 
        "available_bike_stands", "mechanical_bikes", "electrical_bikes",
        "latitude", "longitude", "status", "last_update", "address", "banking", 
        "bonus", "connected", "overflow"
    ]
    
    # Liste pour stocker les données
    data = []
    
    for station_number, station_name in station_data.items():
        # Récupérer les détails de chaque station
        station_details = get_station_details(api_key, contract_name, station_number)
        if station_details:
            row = [
                station_details.get("number", ""),
                station_details.get("name", ""),
                station_details["totalStands"]["availabilities"].get("bikes", 0),
                station_details["totalStands"]["availabilities"].get("stands", 0),
                station_details["totalStands"]["availabilities"].get("mechanicalBikes", 0),
                station_details["totalStands"]["availabilities"].get("electricalBikes", 0),
                station_details["position"].get("latitude", ""),
                station_details["position"].get("longitude", ""),
                station_details.get("status", ""),
                station_details.get("lastUpdate", ""),
                station_details.get("address", ""),
                station_details.get("banking", ""),
                station_details.get("bonus", ""),
                station_details.get("connected", ""),
                station_details.get("overflow", "")
            ]
            data.append(row)
    
    # Création d'un DataFrame pandas
    df = pd.DataFrame(data, columns=headers)
    
    # Sauvegarde en fichier Excel
    df.to_excel(filepath, index=False)
    
    print(f"Données sauvegardées dans le fichier {filepath}")

def get_gbfs_data_and_save_to_excel(city_name):
        url = f"https://api.cyclocity.fr/contracts/{city_name}/gbfs/station_status.json"
        try:
            response = requests.get(url)
            response.raise_for_status()
            gbfs_data = response.json()
            print(f"Données GBFS pour {city_name}: {gbfs_data}")
            # Extraire les données des stations
            stations = gbfs_data.get("data", {}).get("stations", [])
            if not stations:
                print(f"Aucune donnée trouvée pour la ville {city_name}.")
                return

            # Convertir les données en DataFrame
            df = pd.DataFrame(stations)

            # Créer le dossier "gbfs_data" s'il n'existe pas
            os.makedirs("gbfs_data", exist_ok=True)

            # Ajouter la date et l'heure actuelles au nom du fichier
            now = datetime.now()
            date_str = now.strftime("%Y_%m_%d_%H_%M")  # Format : annee_mois_jour_heure_minute

            # Sauvegarder les données dans un fichier Excel
            filename = os.path.join("gbfs_data", f"gbfs_{city_name}_{date_str}.xlsx")
            df.to_excel(filename, index=False)
            print(f"Données GBFS sauvegardées dans le fichier {filename}")
        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la requête GBFS pour {city_name}: {e}")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des données GBFS pour {city_name}: {e}")



# URL de l'API JCDecaux pour récupérer tous les contrats
contracts_url = "https://api.jcdecaux.com/vls/v3/contracts"
api_key = "412c37eac090528b3f24fe5843badacb6f3e907f"  # Remplacez par votre clé API

print("Liste des contrats disponibles :")
get_api_data(contracts_url, api_key)

# Récupérer les stations pour Marseille et Amiens
marseille_stations = get_station_names(api_key, "Marseille")
amiens_stations = get_station_names(api_key, "Amiens")
toulouse_stations = get_station_names(api_key, "Toulouse")

while(True) :
    start_time = time.time()
# Sauvegarder les données dans un fichier CSV pour Marseille et Amiens
    now = datetime.now()
    date_str = now.strftime("%Y_%m_%d_%H_%M")  # Format : annee_mois_jour_heure_minute
    
    # Exemple d'utilisation pour Lyon
    get_gbfs_data_and_save_to_excel("marseille")

    if marseille_stations:
        save_station_data_to_xlsx(marseille_stations, "Marseille", api_key, date_str)

   # if toulouse_stations:
    #    save_station_data_to_csv(toulouse_stations, "Toulouse", api_key, date_str)

    if amiens_stations:
        save_station_data_to_xlsx(amiens_stations, "Amiens", api_key, date_str)
        

    time.sleep(max(0, 60 - (time.time() - start_time)))