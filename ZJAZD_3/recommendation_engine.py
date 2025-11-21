"""
Recommendation Engine
Autorzy: Marek Lewańczyk, Katarzyna Kasperek

Uruchomienie:
    python recommendation_engine.py "Imię Nazwisko"

Etapy:
1) extract_unique_titles
2) tmdb_mapping
3) merge_ratings_with_tmdb
4) fetch_tmdb_details
5) build_features
6) recommend_for_user

Wymagania:
- Python 3.9+
- requirements.txt
- w kodzie udostępniam klucz API, wygenerowany na potrzeby zajęć będzie on aktywny do 22.11.2025 roku

Dokładna instrukcja opisana jest w pliku README.md
"""

import sys
import time

from scripts import (extract_unique_titles, tmdb_mapping, merge_ratings_with_tmdb, fetch_tmdb_details, build_features, recommend_for_user)

def step(name: str, func):
    """Helper — ładny logging i kontrola przepływu."""
    print(f"\n=== [STEP] {name} ===")
    start = time.time()
    func()
    end = time.time()
    print(f"[DONE] {name} ({end - start:.2f}s)")


def main():
    if len(sys.argv) < 2:
        print("Użycie: python recommendation_engine.py \"Imię Nazwisko\"")
        sys.exit(1)

    target_user = sys.argv[1]
    print(f"\n=== URUCHAMIAM CAŁY PIPELINE DLA UŻYTKOWNIKA: {target_user} ===\n")

    # 1. Unikalne tytuły
    step("Extract unique titles", extract_unique_titles.main)

    # 2. TMDB mapping
    step("Map titles to TMDB", tmdb_mapping.main)

    # 3. Join ratingów z TMDB ID
    step("Join ratings with TMDB", merge_ratings_with_tmdb.main)

    # 4. Pobieranie pełnych danych TMDB
    step("Fetch TMDB details", fetch_tmdb_details.main)

    # 5. Ekstrakcja cech
    step("Extract features", build_features.main)

    # 6. Generowanie rekomendacji dla użytkownika
    print("\n=== [GENERATING RECOMMENDATIONS] ===\n")
    recommend_for_user.run_for_user(target_user)   # specjalna funkcja do wywołania


if __name__ == "__main__":
    main()