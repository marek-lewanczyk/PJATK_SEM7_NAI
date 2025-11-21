import csv      # Obsługa CSV (ratingi użytkowników)
import json     # Obsługa JSON (features.jsonl)
from collections import defaultdict  # Słownik z domyślną wartością 0.0

# Ścieżki wejściowe
RATINGS_FILE = "data/ratings_with_tmdb.csv"  # Połączone ratingi + TMDB ID
FEATURES_FILE = "data/features.jsonl"        # Ekstrahowane cechy filmów/seriali

TARGET_USER = "Imię Nazwisko"

# Thresholdy do klasyfikacji pozytywnej i negatywnej
POS_THRESHOLD = 8    # Oceny ≥ 8 traktujemy jako “lubię”
NEG_THRESHOLD = 4    # Oceny ≤ 4 traktujemy jako “nie lubię”


def load_features(path=FEATURES_FILE):
    """
    Wczytuje cechy filmów/seriali z pliku JSON Lines.

    Parametry:
        path (str): Ścieżka do pliku features.jsonl.

    Zwraca:
        dict: Mapowanie tmdb_id -> pełny rekord cech.
    """
    feats = {}
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()               # Usuwamy whitespace
            if not line:                      # Skip pustych linii
                continue
            rec = json.loads(line)            # Parsujemy JSONL do dict
            feats[rec["tmdb_id"]] = rec       # Rejestracja rekordów według klucza tmdb_id
    return feats


def load_user_ratings(user_name, path=RATINGS_FILE):
    """
    Wczytuje ratingi pojedynczego użytkownika i klasyfikuje je na:
    - pozytywne (ocena ≥ POS_THRESHOLD)
    - negatywne (ocena ≤ NEG_THRESHOLD)
    - all_rated (wszystkie obejrzane elementy)

    Parametry:
        user_name (str): Imię i nazwisko użytkownika.
        path (str): Ścieżka do CSV z ratingami.

    Zwraca:
        positives (set), negatives (set), all_rated (set), ratings_map (dict)
    """
    positives = set()     # TMDB ID pozytywnych ocen
    negatives = set()     # TMDB ID negatywnych ocen
    all_rated = set()     # TMDB ID wszystkich ocenionych
    ratings_map = {}      # tmdb_id -> (title, rating)

    with open(path, newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)  # Odczyt CSV na dict per row

        for row in reader:
            if row["user_name"] != user_name:   # Filtr po użytkowniku
                continue

            tmdb_id = row["tmdb_id"]            # Pobranie TMDB ID
            title = row["movie_title"]          # Tytuł filmu/serialu
            rating = float(row["rating"])       # Konwersja ratingu do float

            all_rated.add(tmdb_id)              # Agregacja wszystkich ocen
            ratings_map[tmdb_id] = (title, rating)

            # Klasyfikacja pozytywna
            if rating >= POS_THRESHOLD:
                positives.add(tmdb_id)

            # Klasyfikacja negatywna
            elif rating <= NEG_THRESHOLD:
                negatives.add(tmdb_id)

    return positives, negatives, all_rated, ratings_map


def build_tag_weights(positives, negatives, features_by_id):
    """
    Buduje wektor wag cech użytkownika w oparciu o jego historię ocen.

    Zasada:
    - Każdy pozytywny film dodaje +1 do jego gatunków i słów kluczowych.
    - Każdy negatywny film odejmuje -1 od jego gatunków i słów kluczowych.

    Parametry:
        positives (set): TMDB ID filmów/seriali ocenionych pozytywnie.
        negatives (set): TMDB ID filmów/seriali ocenionych negatywnie.
        features_by_id (dict): Mapowanie tmdb_id -> cechy.

    Zwraca:
        dict: (tag_type, tag_name) -> weight
    """
    weights = defaultdict(float)  # Domyślnie 0.0

    # Przetwarzanie pozytywów
    for tmdb_id in positives:
        f = features_by_id.get(tmdb_id)
        if not f:
            continue

        # Wzmacniamy gatunki
        for g in f["genres"]:
            weights[("genre", g)] += 1.0

        # Wzmacniamy słowa kluczowe
        for k in f["keywords"]:
            weights[("kw", k)] += 1.0

    # Przetwarzanie negatywów
    for tmdb_id in negatives:
        f = features_by_id.get(tmdb_id)
        if not f:
            continue

        # Osłabiamy gatunki
        for g in f["genres"]:
            weights[("genre", g)] -= 1.0

        # Osłabiamy słowa kluczowe
        for k in f["keywords"]:
            weights[("kw", k)] -= 1.0

    return weights


def score_item(tmdb_id, features_by_id, weights):
    """
    Oblicza wynik rekomendacji elementu na podstawie cech i wag użytkownika.

    Parametry:
        tmdb_id (str): ID filmu/serialu.
        features_by_id (dict): Mapowanie ID -> cechy.
        weights (dict): Wektor wag użytkownika.

    Zwraca:
        float | None: Wynik, albo None jeśli brak cech.
    """
    f = features_by_id.get(tmdb_id)
    if not f:
        return None

    s = 0.0  # Suma score’u

    # Dopasowanie gatunków
    for g in f["genres"]:
        s += weights.get(("genre", g), 0.0)

    # Dopasowanie słów kluczowych
    for k in f["keywords"]:
        s += weights.get(("kw", k), 0.0)

    return s


def main():
    """
    Główna procedura:
    1. Wczytaj cechy TMDB.
    2. Wczytaj ratingi użytkownika.
    3. Zbuduj wektor wag użytkownika.
    4. Oceń wszystkie nieoglądane pozycje.
    5. Zwróć top 5 rekomendacji i top 5 anty-rekomendacji.
    """

    features_by_id = load_features()                            # Dane TMDB
    positives, negatives, all_rated, ratings_map = load_user_ratings(TARGET_USER)

    print(f"User: {TARGET_USER}")
    print(f"Positives (>= {POS_THRESHOLD}): {len(positives)}")
    print(f"Negatives (<= {NEG_THRESHOLD}): {len(negatives)}")

    weights = build_tag_weights(positives, negatives, features_by_id)

    # Kandydaci do rekomendacji → wszystkie elementy, których user nie ocenił
    candidates = [tid for tid in features_by_id.keys() if tid not in all_rated]

    scored = []
    for tid in candidates:
        s = score_item(tid, features_by_id, weights)  # Liczymy score
        if s is None:
            continue
        scored.append((tid, s))

    if not scored:
        print("Brak kandydatów do oceny.")
        return

    scored.sort(key=lambda x: x[1], reverse=True)  # Sort malejąco po wyniku

    top5 = scored[:5]      # Najlepsze propozycje
    flop5 = scored[-5:]    # Najgorsze propozycje

    print("\n=== TOP 5 REKOMENDACJE ===")
    for tid, s in top5:
        f = features_by_id[tid]
        print(f"- {f['title']} (id={tid}) | score={s:.2f} | genres={f['genres']}")

    print("\n=== TOP 5 ANTYREKOMENDACJE ===")
    for tid, s in flop5:
        f = features_by_id[tid]
        print(f"- {f['title']} (id={tid}) | score={s:.2f} | genres={f['genres']}")


def run_for_user(user_name: str):
    global TARGET_USER
    TARGET_USER = user_name
    main()

# Standardowy punkt wejścia
if __name__ == "__main__":
    main()