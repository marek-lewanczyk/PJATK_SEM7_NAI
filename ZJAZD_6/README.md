# Detektor flag (PL / RU / UA)

Krótki przewodnik po uruchomieniu i strojenia detektora flag w katalogu `ZJAZD_6`.

## Co robi ten skrypt

Program znajduje prostokątne obszary (kandydaci) na klatce wideo, segmentuje
kolory w przestrzeni HSV i na podstawie udziału kolorów w trzech poziomych
segmentach klasyfikuje obszar jako flagę Polski (PL), Rosji (RU) lub Ukrainy (UA).

Główne pliki:
- `main.py` — detektor i pętla odczytu wideo (główne API).
- `test_smoke.py` — prosty test generujący syntetyczne flagi i sprawdzający klasyfikator.

## Wymagania

Minimalne biblioteki (Python 3.8+):
- numpy
- opencv-python

Instalacja (zalecane w wirtualnym środowisku):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python
```

## Uruchomienie

Uruchom detektor na kamerze (domyślnie `source=0`):

```bash
python3 main.py
```

Uruchom na pliku wideo:

```bash
python3 main.py --source path/to/video.mp4
```

Zatrzymanie: naciśnij `q` lub `Esc` w oknie z podglądem.

## Przykładowe uruchomienie

https://drive.google.com/file/d/1sUgyapXPKbLW8Ah-e7dCP3CXHZgcIlt5/view?usp=sharing

## Kluczowe progi i jak je stroić

W `main.py` znajdziesz stałe konfiguracyjne (przykład i sugestie jak je zmieniać):

- `MIN_CONTOUR_AREA = 3000`
  - Co to: minimalna powierzchnia konturu (w pikselach) aby był rozważany jako kandydat.
  - Strojenie: dla wyższych rozdzielczości lub gdy flagi są duże, zwiększ (np. 10000).
             dla małych flag zmniejsz (np. 1000).

- `MIN_RECT_WIDTH = 80`, `MIN_RECT_HEIGHT = 50`
  - Co to: minimalne wymiary prostokąta w pikselach.
  - Strojenie: jeśli wykrywasz małe flagi, zmniejsz wartości; jeśli masz dużo drobnych
    artefaktów, zwiększ, by je odfiltrować.

- `MIN_ASPECT = 1.4`, `MAX_ASPECT = 3.0`
  - Co to: zakres dopuszczalnych proporcji szerokości do wysokości (w typowych flagach szerokość > wysokość).
  - Strojenie: dopasuj jeśli Twoje flagi mają niestandardowe proporcje.

- `STRIPE_MIN = 0.18`
  - Co to: minimalny udział pikseli koloru w segmencie (np. 0.18 = 18%).
  - Strojenie: jeśli flagi są częściowo zasłonięte lub kolory są rozmyte, zmniejsz (np. 0.10).

Uwaga: wartości są empiryczne — testuj na 20–50 przykładowych klatkach i obserwuj False Positives/Negatives.

## Interpretacja wyników

Funkcja `detect_flags` zwraca obiekt `Detection` z polami:
- `label` — etykieta: `PL` / `RU` / `UA`
- `confidence` — heurystyczna pewność (0.0–1.0)
- `bbox` — prostokąt (x, y, w, h) — x,y to lewy górny róg w pikselach; w,h to szerokość i wysokość

W programie wyświetlane jest okno z narysowanym bounding box i etykietą.

