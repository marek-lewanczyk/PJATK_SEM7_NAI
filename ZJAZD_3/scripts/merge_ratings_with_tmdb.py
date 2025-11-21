import csv  # Obsługa plików CSV

# Ścieżki wejściowe/wyjściowe
RATINGS = "data/group_ratings_raw.csv"      # Surowe oceny od użytkowników
MAPPING = "data/tmdb_mapping.csv"           # Mapowanie: tytuł -> TMDB metadata
OUTPUT = "data/ratings_with_tmdb.csv"       # Finalny plik z joinem TMDB + ratingi


def main():
    """
    Łączy dwa datasety:
    1) Surowe oceny użytkowników (group_ratings_raw.csv)
    2) Mapowanie TMDB (tmdb_mapping.csv)

    Cel:
    - Utrzymać tylko te rekordy, które udało się przypiąć do TMDB.
    - Wzbogacić ratingi o tmdb_id + media_type.
    - Wyrzucić rekordy bez dopasowania (drop).

    Pipeline:
    - Wczytanie mappingu tytułów -> {tmdb_id, media_type}
    - Auto-detekcja kolumny user_name (uwzględnia BOM)
    - Join po tytule
    - Eksport csv: user_name, movie_title, rating, tmdb_id, media_type
    """

    # 1. Wczytanie mapowania: tytuł -> (tmdb_id, media_type)
    mapping = {}  # Słownik: {"Tytuł": ("TMDB_ID", "media_type")}

    with open(MAPPING, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)  # CSV parser operujący na nazwach kolumn
        print("MAPPING columns:", reader.fieldnames)  # Debug: jakie kolumny wykryto

        # Heurystyka wyboru nazwy kolumny z tytułem
        if "movie_title" in reader.fieldnames:
            title_col = "movie_title"  # Standardowy przypadek
        elif "movie_title_raw" in reader.fieldnames:
            title_col = "movie_title_raw"  # Alternatywny fallback
        else:
            raise ValueError("Nie znalazłem 'movie_title' ani 'movie_title_raw' w tmdb_mapping.csv")

        # Iteracja po rekordach mappingu
        for row in reader:
            title = (row.get(title_col) or "").strip()         # Normalizacja tytułu
            tmdb_id = (row.get("tmdb_id") or "").strip()       # Pobranie ID TMDB
            media_type = (row.get("media_type") or "").strip() # Pobranie typu (movie/tv)

            if not title or not tmdb_id:  # Skip: brak danych → nie dodajemy
                continue

            mapping[title] = (tmdb_id, media_type)  # Rejestracja rekordu w słowniku

    print(f"Mapping entries: {len(mapping)}")  # Debug: ile tytułów mamy na stanie

    # 2. Join z ratingami
    kept = 0     # Licznik rekordów finalnych
    dropped = 0  # Licznik odrzuconych (brak dopasowania TMDB)

    with open(RATINGS, newline="", encoding="utf-8-sig") as fin, \
         open(OUTPUT, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)  # Parser dla ratingów
        print("RATINGS columns:", reader.fieldnames)  # Debug: wykryte kolumny

        # Auto-detekcja nazwy kolumny użytkownika – na wypadek BOM lub innej nazwy
        possible_user_cols = ["user_name", "username", "user"]  # Lista możliwych wariantów
        user_col = None

        for c in reader.fieldnames:  # Analiza wszystkich kolumn wejściowych
            if c in possible_user_cols:
                user_col = c  # Idealne trafienie
                break
            # Dodatkowy fallback – case-insensitive i usuwanie spacji
            if c.strip().lower() in [p.lower() for p in possible_user_cols]:
                user_col = c
                break

        if not user_col:
            raise ValueError(
                f"Nie znalazłem kolumny użytkownika w group_ratings_raw.csv. Mam: {reader.fieldnames}"
            )

        # Walidacja obecności kolumny movie_title
        if "movie_title" not in reader.fieldnames:
            raise ValueError("Brak kolumny 'movie_title' w group_ratings_raw.csv")

        # Finalny layout wyjścia
        fieldnames = ["user_name", "movie_title", "rating", "tmdb_id", "media_type"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()  # Nagłówek outputu

        # Iteracja po ratingach
        for row in reader:
            title = (row.get("movie_title") or "").strip()  # Normalizacja tytułu

            # Jeśli tytułu nie ma w mappingu – drop
            if title not in mapping:
                dropped += 1
                continue

            tmdb_id, media_type = mapping[title]  # Pobranie dopasowania TMDB

            # Zapis wzbogaconego rekordu
            writer.writerow({
                "user_name": row[user_col],  # Zidentyfikowana kolumna usera
                "movie_title": title,
                "rating": row["rating"],
                "tmdb_id": tmdb_id,
                "media_type": media_type,
            })

            kept += 1  # Zliczamy utrzymane rekordy

    # Log podsumowujący
    print(f"Kept: {kept}, dropped (no tmdb_id): {dropped}")


# Standardowy punkt wejścia w aplikacji Python
if __name__ == "__main__":
    main()  # Odpalamy end-to-end pipeline