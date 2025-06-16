import requests

def get_timezone_from_city(city):
    # Étape 1 : Trouver coordonnées depuis nom de ville
    search_url = f"http://api.geonames.org/searchJSON?q={city}&maxRows=1&username=dahmanei"
    search_response = requests.get(search_url).json()
    
    if "geonames" not in search_response or not search_response["geonames"]:
        raise ValueError(f"Ville introuvable : {city}")
    
    lat = search_response["geonames"][0]["lat"]
    lng = search_response["geonames"][0]["lng"]

    # Étape 2 : Utiliser coordonnées pour obtenir le timezone réel
    tz_url = f"http://api.geonames.org/timezoneJSON?lat={lat}&lng={lng}&username=dahmanei"
    tz_response = requests.get(tz_url).json()

    if "timezoneId" not in tz_response:
        raise ValueError("Impossible d'obtenir le fuseau horaire.")

    return tz_response["timezoneId"]

def get_time_from_timezone(timezone):
    url = f"https://timeapi.io/api/Time/current/zone?timeZone={timezone}"
    response = requests.get(url).json()

    if "dateTime" not in response:
        raise ValueError("Impossible de récupérer l'heure.")
    
    return response["dateTime"]

def main():
    city = input("Entrez le nom d'une ville : ").strip()
    
    try:
        timezone = get_timezone_from_city(city)
        time = get_time_from_timezone(timezone)
        print(f"🕒 Heure actuelle à {city.title()} ({timezone}) : {time}")
    except Exception as e:
        print("❌ Erreur :", e)

if __name__ == "__main__":
    main()
