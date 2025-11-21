import csv      # Obsługa plików CSV
import json     # Serializacja JSON
import time     # Throttling, żeby nie ubić rate-limitów TMDB
import requests # HTTP client do komunikacji z TMDB API

# Pliki I/O
INPUT = "data/ratings_with_tmdb.csv"   # Wejście: ratingi połączone z TMDB ID
OUTPUT = "data/tmdb_details.jsonl"     # Wyjście: linia JSON per rekord (JSON Lines)

# TMDB API konfiguracja
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5YTJmMjY0NmJmMGZmMzZhMGQzZTdiZWE1YmNjMTYzNyIsIm5iZiI6MTc2MzQxNzk0My4zNDMwMDAyLCJzdWIiOiI2OTFiOWY1NzU5OTYzYTgzNjE2ZDk2M2EiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.y7Q-WxsymQucwDpaMna9_32bp_2XWuzGYd_TjuZUsdI"
BASE_URL = "https://api.themoviedb.org/3"

HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",  # Token Bearer do autoryzacji
    "accept": "application/json",             # Oczekujemy JSON
}


def fetch_movie(id_):
    """
    Pobiera pełne dane o filmie z TMDB.

    Parametry:
        id_ (str): TMDB ID filmu.

    Zwraca:
        dict – kompletny payload filmu (keywords, credits itp.)
    """

    params = {
        "language": "pl-PL",                  # Preferowany język odpowiedzi
        "append_to_response": "keywords,credits",  # Rozszerzenie o dodatkowe sekcje
    }

    r = requests.get(
        f"{BASE_URL}/movie/{id_}",            # Endpoint /movie/{id}
        headers=HEADERS,
        params=params
    )
    r.raise_for_status()                      # Rzuca wyjątek przy błędzie HTTP
    return r.json()                           # Zwraca pełny JSON payload


def fetch_tv(id_):
    """
    Pobiera pełne dane o serialu z TMDB.

    Parametry:
        id_ (str): TMDB ID serialu.

    Zwraca:
        dict – kompletny payload serialu (keywords, credits itp.)
    """

    params = {
        "language": "pl-PL",                   # Preferowany język odpowiedzi
        "append_to_response": "keywords,credits",
    }

    r = requests.get(
        f"{BASE_URL}/tv/{id_}",                # Endpoint /tv/{id}
        headers=HEADERS,
        params=params
    )
    r.raise_for_status()
    return r.json()


def main():
    """
    Proces ETL:
    1. Skonsumuj ratings_with_tmdb.csv.
    2. Wyciągnij unikalne pary (tmdb_id, media_type).
    3. Dla każdego ID pobierz pełne dane z TMDB (movie lub tv).
    4. Zapisz każdy rekord w formacie JSON Lines (1 JSON per linia).
    """

    # 1. Ekstrakcja unikalnych itemów TMDB z pliku ratingów
    items = {}  # Dict: { "tmdb_id": "movie"/"tv" }

    with open(INPUT, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)          # Parser CSV z ratingów

        for row in reader:
            tmdb_id = (row.get("tmdb_id") or "").strip()       # Pobranie ID
            media_type = (row.get("media_type") or "").strip() # Pobranie typu (movie/tv)

            if not tmdb_id or not media_type:  # Skip niepoprawnych rekordów
                continue

            items[tmdb_id] = media_type        # Deduplicate – ostatnia wartość nadpisuje

    print(f"Unique TMDB items: {len(items)}")  # Debug: liczba zapytań do wykonania

    # 2. Pobieranie kompletnej zawartości z TMDB dla każdej unikalnej pozycji
    with open(OUTPUT, "w", encoding="utf-8") as fout:
        for tmdb_id, media_type in items.items():

            try:
                # Select endpoint: movie / tv
                if media_type == "movie":
                    data = fetch_movie(tmdb_id)
                elif media_type == "tv":
                    data = fetch_tv(tmdb_id)
                else:
                    # Nieznany typ – pomijamy
                    print(f"[SKIP] {tmdb_id} unknown media_type={media_type}")
                    continue

                # Budowa rekordu wynikowego
                record = {
                    "tmdb_id": tmdb_id,
                    "media_type": media_type,
                    "data": data,                     # Kompletny payload: keywords, credits itd.
                }

                # Zapis jako JSON Lines – 1 linia = 1 rekord
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"[OK] {tmdb_id} ({media_type})")  # Logging sukcesu

                # Delikatny throttle — zapobiega rate-limitowaniu
                time.sleep(0.1)

            except Exception as e:
                # Nie zatrzymujemy procesu przy pojedynczym błędzie
                print(f"[ERROR] {tmdb_id}: {e}")
                continue


# Standardowy punkt wejścia aplikacji
if __name__ == "__main__":
    main()