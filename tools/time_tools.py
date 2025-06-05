from crewai.tools import tool
import requests
import pytz
from datetime import datetime
import dateparser
import re

@tool("validate_datetime_in_city")
def validate_datetime_in_city(city: str, date_str: str) -> str:
    """
    Même principe qu'avant, mais on re-vérifie la date une fois l’année suivante injectée.
    On essaie d'abord de parser date_str comme ISO. Si ça échoue, on retombe sur dateparser.
    """
    try:
        # 1) Recherche des coordonnées
        search_url = f"http://api.geonames.org/searchJSON?q={city}&maxRows=1&username=dahmanei"
        search_response = requests.get(search_url, timeout=5).json()
        if not search_response.get("geonames"):
            return f"Ville introuvable : {city}"
        lat = search_response["geonames"][0]["lat"]
        lng = search_response["geonames"][0]["lng"]

        # 2) Récupération du fuseau horaire
        tz_url = f"http://api.geonames.org/timezoneJSON?lat={lat}&lng={lng}&username=dahmanei"
        tz_response = requests.get(tz_url, timeout=5).json()
        timezone = tz_response.get("timezoneId")
        if not timezone:
            return "Impossible de récupérer le fuseau horaire."

        tz = pytz.timezone(timezone)
        now = datetime.now(tz)

        # ────────────────
        # 3) Tenter ISO-8601
        # ────────────────
        try:
            parsed = datetime.fromisoformat(date_str)
            if parsed.tzinfo is None:
                candidate = tz.localize(parsed)
            else:
                candidate = parsed.astimezone(tz)

            if candidate < now:
                return f"La date {date_str} est déjà passée à {city.title()} ({timezone})."
            else:
                return f"La date {date_str} est dans le futur à {city.title()} ({timezone})."

        except ValueError:
            # date_str n’était pas en format ISO valide → on retombe sur le parsing libre
            pass

        # ────────────────
        # 4) FALLBACK : votre parsing “naturel” existant
        # ────────────────
        # Si date_str ne contient pas d’année, on ajoute par défaut 2025
        year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
        if not year_match:
            date_str_for_parse = f"{date_str} 2025"
        else:
            date_str_for_parse = date_str

        settings = {
            "DATE_ORDER": "DMY",
            "PREFER_DATES_FROM": "future",
            "RELATIVE_BASE": now
        }
        parsed_nl = dateparser.parse(date_str_for_parse, languages=["fr"], settings=settings)
        if parsed_nl is None:
            return f"Format de date invalide : '{date_str}'. Utilisez un ISO (YYYY-DD-MMThh:mm:ss)."

        candidate_nl = tz.localize(parsed_nl)

        # Si cette date est déjà passée, essayer l’année suivante
        if candidate_nl < now:
            if not year_match:
                # Remplacer “2025” par “2026”
                date_str_for_parse_next = re.sub(r"2025$", "2026", date_str_for_parse)
                parsed_next = dateparser.parse(date_str_for_parse_next, languages=["fr"], settings=settings)
                if parsed_next:
                    candidate_next = tz.localize(parsed_next)
                    if candidate_next < now:
                        iso = candidate_next.isoformat()
                        return f"La date {iso} est déjà passée à {city.title()} ({timezone})."
                    else:
                        iso = candidate_next.isoformat()
                        return f"La date {iso} est dans le futur à {city.title()} ({timezone})."
                # si on ne parvient pas à parser en 2026 non plus, on considère que c’est déjà passé
                iso = candidate_nl.isoformat()
                return f"La date {iso} est déjà passée à {city.title()} ({timezone})."
            else:
                # Si l’utilisateur avait donné explicitement une année (ex. 2023), on renvoie “déjà passée”
                iso = candidate_nl.isoformat()
                return f"La date {iso} est déjà passée à {city.title()} ({timezone})."
        else:
            # Candidat n’est pas dans le passé
            iso = candidate_nl.isoformat()
            return f"La date {iso} est dans le futur à {city.title()} ({timezone})."

    except Exception as e:
        return f"Erreur lors de la validation : {str(e)}"
