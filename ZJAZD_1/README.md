# Arithmetic Tug-of_War (gra o sumie zerowej)
# Opis
Jest to gra dwuosobowa, deterministyczna o sumie zerowej, w której gracze "przeciągają linę" na osi liczbowej od -K do +K.
Player1 dąży do uzyskania pozycji +K a Player2 do -K.

Gra została zaimplementowania w Pythonie z wykorzystaniem biblioteki easyAi i algorytmu Negmax (wariant Minimax), który pozwala sztucznej inteligencji analizować ruchy przeciwnika i dobierać najlepszą strategię.

# Wymagania
- Python 3.9 lub nowszy
- pip
- Biblioteka "easyAI"

# Instalacja
Sprawdź czy posiadasz Python

MacOS

`code python3 --version`

Jeśli brak - zainstaluj przez Homebrew

`brew install python`

Sprawdź równiez pip

`pip3 --version`

Następnie zainstaluj blibliotekę easyAI

`pip3 install easyAI`

Windows

1. Pobierz instalator ze strony https://www.python.org/downloads/
2. Podczas instalacji ZAZNACZ opcję 'Add Python to PATH'
3. Sprawdź instalację
```
python --version
pip --version
```
Następnie zainstaluj bibliotekę easyAI

`pip install easyAI`

Linux
```
sudo apt update
sudo apt install python3 python3-pip -y
```
Następnie zainstaluj bibliotekę easyAI

`pip3 install easyAI`

# Uruchomienie gry

1. Otwórz terminal i przejdź do katalogu projektu

2. Wybierz tryb w którym chcesz uruchomić grę. Dostępne są dwa tryby: człowiek vs AI oraz AI vs AI

- tryb człowiek vs AI

`python3 main.py --mode human_vs_ai`

- tryb AI vs AI

`python3 main.py --mode ai_vs_ai`

# Dodatkowe opcje uruchomienia gry

- graniczna wartość (zasięg gry)
  - --k
- dozwolone wartości ruchów
  - --moves
- głębokość analizy AI
  - --depth

# Zakończenie gry

Gra kończy się w momencie, gdy znacznik osiągnie wartość 

- +K - wygrywa Player1
- -K - wygrywa Player2



