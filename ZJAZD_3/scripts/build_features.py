import json  # Obsługa serializacji/deseializacji JSON

# Pliki wejścia/wyjścia
INPUT = "data/tmdb_details.jsonl"   # Wejście: JSONL z pełnymi danymi TMDB
OUTPUT = "data/features.jsonl"      # Wyjście: JSONL z wyekstrahowanymi cechami


def extract_year(date_str: str | None) -> int | None:
    """
    Ekstrahuje rok (YYYY) z daty w formacie 'YYYY-MM-DD'.

    Parametry:
        date_str (str | None): String daty lub None.

    Zwraca:
        int | None – Rok jako liczba, albo None jeśli data niepoprawna.
    """
    if not date_str:           # Brak wartości → zwracamy None
        return None
    try:
        return int(date_str.split("-")[0])  # Parsujemy rok z pierwszego segmentu
    except Exception:
        return None            # Błąd parsowania → zwracamy None


def main():
    """
    Pipeline transformacyjny:
    - Przetwarza każdy rekord JSONL z pełnymi metadanymi TMDB.
    - Ekstrahuje wybrane cechy (genres, keywords, year, popularity itd.).
    - Dla filmów i seriali uwzględnia różnicę w strukturze keywords.
    - Zapisuje rekord wynikowy w formacie JSON Lines.

    Finalny output to "feature store" pod dalsze modelowanie ML.
    """

    with open(INPUT, encoding="utf-8") as fin, \
         open(OUTPUT, "w", encoding="utf-8") as fout:

        count = 0  # Licznik przetworzonych rekordów

        for line in fin:
            line = line.strip()           # Czyścimy whitespace
            if not line:                  # Skip pustych linii
                continue

            record = json.loads(line)     # Parsujemy linię JSON
            tmdb_id = record["tmdb_id"]   # Pobieramy ID TMDB
            media_type = record["media_type"]  # Pobieramy media_type
            data = record["data"]         # Pełny payload TMDB

            # 1. GENRES
            genres = [g["name"] for g in data.get("genres", [])]  # Lista nazw gatunków

            # 2. KEYWORDS
            keywords_block = data.get("keywords") or {}  # Film i TV mają różne formaty

            if isinstance(keywords_block, dict):
                # Case: movie → {"keywords": [...]}
                if "keywords" in keywords_block:
                    keywords = [k["name"] for k in keywords_block["keywords"]]

                # Case: tv → {"results": [...]}
                elif "results" in keywords_block:
                    keywords = [k["name"] for k in keywords_block["results"]]

                else:
                    keywords = []
            else:
                keywords = []  # Fallback: niepoprawny format

            # 3. YEAR
            release_date = (
                data.get("release_date")          # Filmy
                or data.get("first_air_date")     # Seriale
            )
            year = extract_year(release_date)  # Ekstrakcja roku

            # 4. METRYKI (vote_average, popularity)
            vote_average = data.get("vote_average")  # Średnia ocen TMDB
            popularity = data.get("popularity")      # Popularność wg TMDB scoring

            # 5. DURATION / RUNTIME
            runtime = data.get("runtime") or None    # Tylko dla filmów

            # 6. KONSTRUKCJA OUTPU
            out = {
                "tmdb_id": tmdb_id,                        # Klucz główny
                "media_type": media_type,                  # movie / tv
                "title": data.get("title") or data.get("name"),  # Film vs serial
                "genres": genres,                          # Lista gatunków
                "keywords": keywords,                      # Lista słów kluczowych
                "year": year,                              # Rok premiery
                "vote_average": vote_average,              # Średnia ocen
                "popularity": popularity,                  # Popularność
                "runtime": runtime,                        # Czas trwania filmu
            }

            # Serializacja do JSONL
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            count += 1  # Aktualizacja licznika

    print(f"Zapisano {count} rekordów do {OUTPUT}")  # Podsumowanie operacji


# Standardowy punkt wejścia
if __name__ == "__main__":
    main()