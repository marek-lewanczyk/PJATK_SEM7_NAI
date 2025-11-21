import csv  # Import standardowej biblioteki CSV do pracy z plikami

INPUT = "data/group_ratings_raw.csv"  # Ścieżka do wejściowego pliku CSV
OUTPUT = "data/unique_titles.csv"  # Ścieżka do wyjściowego pliku z unikalnymi tytułami


def main():
    """
    Procesuje surowy plik group_ratings_raw.csv, ekstrahując unikalne tytuły filmów.

    Pipeline operacyjny:
    1. Odczyt CSV wejściowego z ocenami użytkowników.
    2. Wyciągnięcie wartości z kolumny 'movie_title'.
    3. Deduplication – agregacja tytułów w strukturze typu set.
    4. Sortowanie i eksport do nowego CSV pod nazwą unique_titles.csv.

    Wynik:
    - Plik CSV zawierający jedną kolumnę: movie_title.
    - Każdy wiersz to unikatowy tytuł.
    """
    titles = set()  # Zbiór do gromadzenia unikalnych tytułów

    # Otwieramy plik wejściowy w trybie odczytu
    with open(INPUT, newline="", encoding="utf-8") as fin:  # Kontener pliku
        reader = csv.DictReader(fin, delimiter=",")  # Parser CSV oparty o nazwy kolumn
        for row in reader:  # Iterujemy po każdym wierszu z pliku wejściowego
            t = row["movie_title"].strip()  # Pobieramy tytuł i usuwamy białe znaki
            if t:  # Walidacja, czy tytuł nie jest pusty
                titles.add(t)  # Dodajemy tytuł do zbioru unikalnych wartości

    # Otwieramy plik wyjściowy w trybie zapisu
    with open(OUTPUT, "w", newline="", encoding="utf-8") as fout:  # Kontener pliku
        w = csv.writer(fout, delimiter=",")  # Writer CSV do zapisu danych
        w.writerow(["movie_title"])  # Header w pliku wynikowym
        for t in sorted(titles):  # Sortujemy tytuły alfabetycznie
            w.writerow([t])  # Zapisujemy każdy tytuł jako osobny wiersz


# Punkt wejścia aplikacji
if __name__ == "__main__":  # Warunek uruchomienia bezpośredniego
    main()  # Call to action – odpalamy główny workflow