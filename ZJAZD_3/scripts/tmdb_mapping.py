import csv  # Biblioteka standardowa do obsługi plików CSV
import time  # Do throttlingu wywołań API
import requests  # Do komunikacji HTTP z API TMDB

# Bearer token do TMDB API
TMDB_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI5YTJmMjY0NmJmMGZmMzZhMGQzZTdiZWE1YmNjMTYzNyIsIm5iZiI6MTc2MzQxNzk0My4zNDMwMDAyLCJzdWIiOiI2OTFiOWY1NzU5OTYzYTgzNjE2ZDk2M2EiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.y7Q-WxsymQucwDpaMna9_32bp_2XWuzGYd_TjuZUsdI"

# Endpoint bazowy TMDB
BASE_URL = "https://api.themoviedb.org/3"

# Stałe nagłówki HTTP – autoryzacja + format JSON
HEADERS = {
    "Authorization": f"Bearer {TMDB_TOKEN}",
    "accept": "application/json",
}

# Pliki wejściowe i wyjściowe
INPUT_FILE = "data/unique_titles.csv"  # Źródło – lista unikalnych tytułów
OUTPUT_FILE = "data/tmdb_mapping.csv"  # Wyjście – mapowanie tytułów na wyniki TMDB


def search_tmdb(title: str):
    """
    Wysyła zapytanie do TMDB `/search/multi` w celu znalezienia najlepszego dopasowania do podanego tytułu.

    Parametry:
        title (str): Nazwa filmu/serialu, która ma zostać zmapowana.

    Zwraca:
        dict | None:
            Słownik z kluczowymi metadanymi TMDB (media_type, tmdb_id, original_title,
            release_date, vote_count, popularity) – lub None jeśli brak dopasowań.
    """
    params = {
        "query": title,  # Tekst wyszukiwania w TMDB
        "include_adult": "false",  # Filtr wyłączający treści +18
    }

    r = requests.get(  # Wywołanie HTTP GET
        f"{BASE_URL}/search/multi",
        headers=HEADERS,
        params=params
    )
    r.raise_for_status()  # Walidacja 2xx / rzuca wyjątek w razie błędu

    results = r.json().get("results", [])  # Pobranie listy wyników z JSON
    if not results:  # Jeżeli brak wyników – zwracamy None
        return None

    best = results[0]  # Strategia v1 – bierzemy pierwszy wynik jako "najlepszy"
    return {
        "media_type": best.get("media_type"),  # movie / tv / person
        "tmdb_id": best.get("id"),  # ID w TMDB
        "original_title": best.get("original_title") or best.get("name"),  # Film/serial
        "release_date": best.get("release_date") or best.get("first_air_date"),  # Data premiery
        "vote_count": best.get("vote_count"),  # Liczba głosów
        "popularity": best.get("popularity"),  # Popularność wg TMDB
    }


def main():
    """
    Główny pipeline procesu mapowania:

    1. Odczyt listy unikalnych tytułów z CSV.
    2. Wywołanie TMDB API dla każdego tytułu (search/multi).
    3. Zapis najlepszych dopasowań do pliku `tmdb_mapping.csv`.
    4. Kontrolowany throttling, aby nie uderzać w rate-limit API.

    Finalny output:
        CSV z kolumnami:
        movie_title, tmdb_id, media_type, original_title, release_date, vote_count, popularity
    """

    # Otwieramy plik wejściowy i wyjściowy w jednym kontekście
    with open(INPUT_FILE, newline="", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)  # Odczyt CSV -> dict per row

        fieldnames = [
            "movie_title",
            "tmdb_id",
            "media_type",
            "original_title",
            "release_date",
            "vote_count",
            "popularity",
        ]

        writer = csv.DictWriter(fout, fieldnames=fieldnames)  # Writer z ustalonymi kolumnami
        writer.writeheader()  # Nagłówek outputu

        for row in reader:  # Iteracja po każdej linii wejściowej
            title = row["movie_title"].strip()  # Normalizacja whitespace

            if not title:  # Skip pustych tytułów
                continue

            try:
                result = search_tmdb(title)  # Zapytanie do TMDB dla danego tytułu
            except Exception as e:
                print(f"[ERROR] {title}: {e}")  # Log niepowodzenia
                continue

            if not result:
                print(f"[NO MATCH] {title}")  # Log braku wyników
                continue

            writer.writerow({  # Zapis dopasowania do CSV
                "movie_title": title,
                **result,  # Spread dict result
            })

            time.sleep(0.05)  # Throttling – 50 ms pauzy na request


# Standardowy punkt wejścia skryptu
if __name__ == "__main__":  # Zapobiega auto-uruchomieniu przy imporcie
    main()  # Launch pipeline