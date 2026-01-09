"""
Projekt: Detektor flag (PL / RU / UA) w strumieniu wideo.

Opis:
Ten moduł wykrywa prostokątne obszary (kandydatów) na obrazie, segmentuje
kolory w przestrzeni HSV i rozpoznaje flagi Polski, Rosji i Ukrainy na
podstawie układu poziomych pasów kolorów.

Proste użycie:
  python ZJAZD_6/main.py    # domyślnie używa kamery (source=0)
  python ZJAZD_6/main.py --source path/to/video.mp4

Co jest czym?
  - BGR: format kolorów używany przez OpenCV (Blue, Green, Red).
  - HSV: przestrzeń barw (Hue, Saturation, Value) — ułatwia segmentację kolorów.
  - maska: binarna mapa (0/255) mówiąca, które piksele należą do danego koloru.
  - ROI: region of interest, czyli wycinek obrazu do dalszej analizy.
  - kontur: obwód kształtu wykryty w obrazie; używamy go do znalezienia prostokątów.

Plik jest modularny — główne funkcje to:
  - _hsv_masks: tworzy maski kolorów (white/red/blue/yellow)
  - _find_candidate_rectangles: znajdowanie prostokątnych kandydatów na flagę
  - _score_flag_by_stripes: klasyfikacja ROI na podstawie udziałów kolorów
  - detect_flags: główny wrapper łączący wszystkie kroki

Uwagi dotyczące implementacji:
  - Używamy HSV zamiast BGR dla stabilniejszej segmentacji kolorów przy różnym oświetleniu.
  - Maska dla czerwieni jest łączona z dwóch zakresów, bo hue czerwieni owija się na końcach skali.
  - Operacje morfologiczne (open/close) redukują szum w maskach przed policzeniem udziałów.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np

# HSV_RANGES: zakresy kolorów w przestrzeni HSV
# Używamy HSV zamiast RGB/BGR bo wartość H (hue) jest bardziej odporna
# na zmiany oświetlenia i dzięki temu łatwiej odseparować kolory (np. niebieski, czerwony).
HSV_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    # (dolny HSV), (górny HSV) przechowywane jako tablice uint8 — zgodne z
    # typami oczekiwanymi przez OpenCV dla funkcji cv2.inRange. Dzięki temu
    # unikamy ostrzeżeń typów i mamy przewidywalne zachowanie.
    "white": (np.array((0, 0, 200), dtype=np.uint8), np.array((179, 60, 255), dtype=np.uint8)),
    "red1": (np.array((0, 70, 70), dtype=np.uint8), np.array((10, 255, 255), dtype=np.uint8)),
    "red2": (np.array((170, 70, 70), dtype=np.uint8), np.array((179, 255, 255), dtype=np.uint8)),
    "blue": (np.array((85, 50, 50), dtype=np.uint8), np.array((140, 255, 255), dtype=np.uint8)),
    "yellow": (np.array((18, 70, 70), dtype=np.uint8), np.array((40, 255, 255), dtype=np.uint8)),
}

# Mały kernel morfologiczny używany do odszumiania masek
# cv2.getStructuringElement tworzy strukturalny element (kernel) używany
# w operacjach morfologicznych (np. erozja/dylatacja). Użycie funkcji
# OpenCV zapewnia kompatybilność i łatwą zmianę kształtu kernela.
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# Kernel używany do dylatacji mapy krawędzi przed wyszukiwaniem konturów
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Parametry kontrolujące progi detekcji i filtry geometryczne
MIN_CONTOUR_AREA = 3000
MIN_RECT_WIDTH = 80
MIN_RECT_HEIGHT = 50
MIN_ASPECT = 1.4
MAX_ASPECT = 3.0
MAX_CANDIDATES = 10
STRIPE_MIN = 0.18  # minimalny udział koloru w segmencie, aby uznać go za obecny
# Uwaga: wartości empiryczne. Jeśli pracujesz z obrazami o innej rozdzielczości
# lub innym skalowaniu obiektów, dostosuj te progi:
# - MIN_CONTOUR_AREA: zmniejsz dla mniejszych flag lub zwiększ, aby odfiltrować szum
# - MIN_RECT_WIDTH / MIN_RECT_HEIGHT: minimalne wymiary prostokąta w pikselach
# - MIN_ASPECT / MAX_ASPECT: stosunek szerokości do wysokości typowej flagi
# - STRIPE_MIN: minimalny udział koloru w segmencie (np. 0.18 = 18% pikseli)


@dataclass
class Detection:
    """Wynik detekcji flagi na klatce."""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    # bbox = (x, y, w, h) — x,y to współrzędne lewego górnego rogu w pikselach;
    # w,h to szerokość i wysokość w pikselach. Ułatwia to wizualizację i rysowanie.


def _hsv_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Buduje i zwraca maski binarne HSV dla kolorów używanych w detekcji flag.
    Zwraca słownik mapujący nazwę koloru -> jednokanałowa maska uint8.
    Stosuje operacje morfologiczne, żeby zredukować szum.
    """
    # cv2.inRange: zwraca binarną maskę (0/255) w miejscu, gdzie piksel mieści się
    # w zadanym zakresie HSV. To podstawowa metoda segmentacji kolorów w OpenCV.
    white = cv2.inRange(hsv, HSV_RANGES["white"][0], HSV_RANGES["white"][1])

    # Red: czerwony jest na końcu i początku skali H w OpenCV, dlatego łączymy dwa zakresy
    r1 = cv2.inRange(hsv, HSV_RANGES["red1"][0], HSV_RANGES["red1"][1])
    r2 = cv2.inRange(hsv, HSV_RANGES["red2"][0], HSV_RANGES["red2"][1])
    # cv2.bitwise_or łączy dwie maski bitowo — dostajemy spójną maskę czerwieni.
    red = cv2.bitwise_or(r1, r2)

    blue = cv2.inRange(hsv, HSV_RANGES["blue"][0], HSV_RANGES["blue"][1])
    yellow = cv2.inRange(hsv, HSV_RANGES["yellow"][0], HSV_RANGES["yellow"][1])

    def clean(mask: np.ndarray) -> np.ndarray:
        # morphologyEx z MORPH_OPEN (erozja potem dylatacja) usuwa małe 'szumy'
        # MORPH_CLOSE (dylatacja potem erozja) wypełnia małe dziury.
        # Tego używamy, żeby maski były bardziej stabilne przed liczeniem udziałów.
        m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)
        return m

    return {
        "white": clean(white),
        "red": clean(red),
        "blue": clean(blue),
        "yellow": clean(yellow),
    }


def _score_flag_by_stripes(roi_bgr: np.ndarray, masks_roi: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Optional[str], float]:
    """
    Klasyfikuje wycinek obrazu (ROI) jako PL / RU / UA na podstawie
    pionowego ułożenia poziomych pasów kolorów.

    - Dzielimy ROI na 3 poziome segmenty (góra/środek/dół).
    - Dla każdego koloru liczymy udział pikseli (coverage) w segmencie.
    - Na podstawie heurystyk (np. white top + red bottom -> PL) przypisujemy
      etykietę oraz pewność.

    Parametr `masks_roi` pozwala przekazać już policzone maski dla ROI.
    Dzięki temu unikamy kosztownej konwersji BGR->HSV i wywołań cv2.inRange
    dla każdego małego prostokąta — zamiast tego obliczamy maski raz dla
    całej klatki i bierzemy wycinki (slicing). W przypadku braku masek
    funkcja obliczy je lokalnie (mechanizm awaryjny).
    """
    if roi_bgr.size == 0:
        return None, 0.0

    h, w = roi_bgr.shape[:2]
    # Wczesne odrzucenie bardzo małych ROI — większość flag jest większa.
    if h < 40 or w < 60:
        return None, 0.0

    # Jeśli nie podano masek, policzemy je lokalnie (wolniejszy tryb).
    if masks_roi is None:
        # cvtColor: konwersja z BGR (wejście OpenCV) do HSV — przed segmentacją kolorów
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        masks = _hsv_masks(hsv)
    else:
        masks = masks_roi

    # Dzielimy flagę na trzy poziome segmenty: top / mid / bottom
    y1 = h // 3
    y2 = 2 * h // 3
    segs = [(0, y1), (y1, y2), (y2, h)]

    # coverage: szybkie oszacowanie ułamka pikseli należących do maski w segmencie
    # coverage = liczba_pikseli_maski / liczba_wszystkich_pikseli_segm. Zakres [0.0, 1.0]
    # Przykład: coverage = 0.2 oznacza, że 20% pikseli w danym segmencie odpowiada kolorowi.
    # Jeśli STRIPE_MIN = 0.18, wymagamy przynajmniej ~18% pokrycia by uznać pas za obecny.
    def coverage(mask: np.ndarray, y_from: int, y_to: int) -> float:
        seg = mask[y_from:y_to, :]
        # maski mają wartości 0 lub 255, więc np.count_nonzero liczy istotne piksele
        # dzielimy przez rozmiar segmnetu (w liczbie elementów) aby dostać ułamek
        if seg.size == 0:
            return 0.0
        return float(np.count_nonzero(seg)) / float(seg.size)

    # cov to słownik: kolor -> [coverage_top, coverage_mid, coverage_bot]
    cov = {c: [coverage(masks[c], a, b) for (a, b) in segs] for c in masks.keys()}

    colors = ["white", "red", "blue", "yellow"]

    def dominant_in_segment(seg_idx: int) -> Tuple[str, float]:
        vals = [(c, cov[c][seg_idx]) for c in colors]
        vals.sort(key=lambda x: x[1], reverse=True)
        return vals[0]

    top = dominant_in_segment(0)
    mid = dominant_in_segment(1)
    bot = dominant_in_segment(2)

    # --- PL (biało-czerwone) ---
    pl_score = 0.0
    if top[0] == "white" and bot[0] == "red" and top[1] > STRIPE_MIN and bot[1] > STRIPE_MIN:
        # prosty sposób skalowania pewności
        pl_score = min(1.0, (top[1] + bot[1]) / 0.9)

    # Jeśli środkowy pas jest wyraźnie niebieski, to osłabiamy pewność PL
    if mid[0] == "blue" and mid[1] > 0.25:
        pl_score *= 0.2

    # --- UA (niebiesko-żółte) ---
    ua_score = 0.0
    if top[0] == "blue" and bot[0] == "yellow" and top[1] > STRIPE_MIN and bot[1] > STRIPE_MIN:
        ua_score = min(1.0, (top[1] + bot[1]) / 0.9)

    # --- RU (białe + niebieskie + czerwone) ---
    ru_score = 0.0
    top_white = cov["white"][0]
    mid_blue = cov["blue"][1]
    bot_red = cov["red"][2]

    if top_white > 0.14 and mid_blue > 0.10 and bot_red > 0.14:
        # blue ma większą wagę
        ru_score = min(1.0, (top_white + 1.8 * mid_blue + bot_red) / 1.6)

    # Jeśli wygląda jak PL, ale w środku prawie nie ma niebieskiego -> osłab RU
    if top[0] == "white" and bot[0] == "red" and mid_blue < 0.08:
        ru_score *= 0.5

    # Priorytet decyzyjny: jeśli RU jest wystarczająco silne, zwracamy RU
    if ru_score >= 0.50:
        return "RU", float(ru_score)

    # W przeciwnym razie wybieramy między PL i UA
    scores = [("PL", pl_score), ("UA", ua_score)]
    scores.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scores[0]

    if best_score < 0.45:
        return None, 0.0

    return best_label, float(best_score)


def _find_candidate_rectangles(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Szuka prostokątnych obszarów (kandydatów na flagę) na podstawie krawędzi
    i konturów. Zwraca listę bbox (x, y, w, h) posortowaną po rozmiarze malejąco.
    """
    h, w = frame_bgr.shape[:2]

    # cvtColor + GaussianBlur: konwersja do odcieni szarości i wygładzenie
    # Obrazy w skali szarości mają jedną warstwę i są szybsze do obliczeń krawędzi
    # GaussianBlur zmniejsza szum, co redukuje fałszywe krawędzie.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny: algorytm detekcji krawędzi — daje binarną mapę krawędzi
    edges = cv2.Canny(gray, 60, 140)
    # Dilatacja: "poszerza" krawędzie tak, aby kontury były ciągłe i łatwiejsze do znalezienia
    # Używamy wcześniej przygotowanego DILATE_KERNEL.
    edges = cv2.dilate(edges, DILATE_KERNEL, iterations=1)

    # findContours: znajduje kształty (kontury) w binarnej mapie krawędzi
    # Zwraca listę konturów, które później filtrujemy po powierzchni i kształcie.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        # contourArea: powierzchnia obszaru ograniczonego konturem
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue

        # arcLength: obwód konturu
        peri = cv2.arcLength(cnt, True)
        # approxPolyDP: przybliża kontur wielokątem o mniejszej liczbie wierzchołków
        # parametr epsilon (0.02*peri) kontroluje dokładność; jeśli zostanie 4 punktów,
        # traktujemy jako prostokąt przybliżony.
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # Szukamy prostokątów (przybliżenie do 4 punktów)
        if len(approx) != 4:
            continue

        # boundingRect: najprostszy prostokątny bbox obejmujący przybliżony kontur
        x, y, rw, rh = cv2.boundingRect(approx)

        # Filtry geometryczne: flaga zwykle "szersza niż wyższa"
        if rw < MIN_RECT_WIDTH or rh < MIN_RECT_HEIGHT:
            continue
        aspect = rw / float(rh)
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue

        # Przycinamy do granic obrazu
        x = max(0, x); y = max(0, y)
        rw = min(rw, w - x); rh = min(rh, h - y)

        rects.append((x, y, rw, rh))

    # Sortujemy kandydatów po polu malejąco i bierzemy kilka największych (limit)
    rects.sort(key=lambda r: r[2] * r[3], reverse=True)
    return rects[:MAX_CANDIDATES]


def detect_flags(frame_bgr: np.ndarray) -> Optional[Detection]:
    """
    Detekcja najlepszej flagi na klatce. Zwraca Detection albo None.

    Optymalizacja: obliczamy maski HSV (biała, czerwona, niebieska, żółta)
    raz dla całej klatki, a następnie dla każdego kandydata przetwarzamy
    jedynie wycinek mask (slicing), co eliminuje wielokrotne konwersje HSV.
    """
    # Gdy klatka jest pusta lub None, przerywamy (bezpieczeństwo)
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    rects = _find_candidate_rectangles(frame_bgr)

    if not rects:
        return None

    # Wstępnie obliczamy maski HSV dla całej klatki (duży zysk wydajności)
    hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    masks_full = _hsv_masks(hsv_full)

    best: Optional[Detection] = None
    H, W = frame_bgr.shape[:2]
    for (x, y, w, h) in rects:
        pad = int(0.05 * min(w, h))  # 5% margines
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(W, x + w + pad)
        y1 = min(H, y + h + pad)
        roi = frame_bgr[y0:y1, x0:x1]

        # Wycinamy fragmenty (ROI) z wstępnie obliczonych masek, aby uniknąć
        # wielokrotnego przeliczania konwersji BGR->HSV i cv2.inRange dla każdego
        # kandydata. Slicing działa bardzo szybko i zwraca widok odpowiedniego regionu.
        masks_roi = {k: masks_full[k][y0:y1, x0:x1] for k in masks_full.keys()}
        label, conf = _score_flag_by_stripes(roi, masks_roi)
        if label is None:
            continue

        det_box = (x0, y0, x1 - x0, y1 - y0)

        # Dla RU staramy się rozszerzyć bbox w górę, żeby uchwycić biały pas
        if label == "RU":
            # przekazujemy wstępnie policzoną maskę bieli, żeby nie przeliczać jej od zera
            det_box = _expand_ru_bbox_by_white(frame_bgr, det_box, white_mask_full=masks_full.get("white"))

        det = Detection(label=label, confidence=conf, bbox=det_box)
        if best is None or det.confidence > best.confidence:
            best = det

    return best


def _expand_ru_bbox_by_white(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    *,
    white_row_ratio_thresh: float = 0.12,
    max_expand_px: int = 250,
    white_mask_full: Optional[np.ndarray] = None,
) -> Tuple[int, int, int, int]:
    """
    Rozszerza bbox w górę, aby objąć biały pasek (typowy dla flagi RU).

    Przyjmuje opcjonalnie `white_mask_full` (jednokanałowa maska tej samej
    wielkości co `frame_bgr`) — jeśli jest dostępna, zostanie użyta zamiast
    przeliczania masek HSV.

    Funkcja działa bez zmiany szerokości bbox; jedynie przesuwa górną krawędź.
    """
    x, y, w, h = bbox
    H, W = frame_bgr.shape[:2]

    if w <= 0 or h <= 0:
        return bbox

    if white_mask_full is None:
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        white_mask = _hsv_masks(hsv_full)["white"]
    else:
        white_mask = white_mask_full

    x0 = max(0, x)
    x1 = min(W, x + w)
    if x1 <= x0:
        return bbox

    # Szukamy w górę od y, zatrzymując się gdy nie ma już wystarczającej ilości bieli
    y_top = y
    limit = max(0, y - max_expand_px)
    # iterujemy w górę; wektoryzacja nie daje korzyści, bo często zatrzymujemy pętlę wcześnie
    for yy in range(y - 1, limit - 1, -1):
        row = white_mask[yy:yy + 1, x0:x1]
        if row.size == 0:
            break
        white_ratio = float(np.count_nonzero(row)) / float(row.size)
        if white_ratio < white_row_ratio_thresh:
            break
        y_top = yy

    new_y = max(0, y_top)
    new_h = (y + h) - new_y
    return (x, new_y, w, new_h)


def draw_detection(frame_bgr: np.ndarray, det: Detection) -> None:
    """Rysuje bbox i tekst informacyjny na klatce."""
    x, y, w, h = det.bbox
    cx = x + w // 2
    cy = y + h // 2

    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(frame_bgr, (cx, cy), 4, (0, 255, 0), -1)

    text = f"{det.label}  conf={det.confidence:.2f}  bbox=({x},{y},{w},{h})  center=({cx},{cy})"
    # Tekst wypisujemy ponad bbox, ale upewniamy się, że mieści się w obrazie
    cv2.putText(frame_bgr, text, (max(10, x), max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)



def parse_args() -> argparse.Namespace:
    """Parsuje argumenty linii poleceń.

    Zwraca Namespace z polem `source` ("0" dla kamery lub ścieżka do pliku wideo).
    """
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 dla kamery albo ścieżka do pliku wideo")
    return p.parse_args()


def main() -> None:
    """Główny punkt wejścia.

    Otwiera źródło wideo, iteruje po klatkach i pokazuje detekcje na żywo.
    """
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # VideoCapture: interfejs do odczytu klatek z kamery / pliku wideo
    # Konstruktor przyjmuje indeks kamery (np. 0) lub ścieżkę do pliku.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć źródła wideo: {args.source}")

    while True:
        ok, frame = cap.read()
        # Jeśli odczyt klatki nie powiódł się (koniec pliku, błąd kamery itp.),
        # przerywamy pętlę. Dodatkowo sprawdzamy, czy frame nie jest None i ma
        # rozmiar > 0, by uniknąć błędów przy późniejszym przetwarzaniu.
        if not ok or frame is None or frame.size == 0:
            break

        # detect_flags: najważniejsza funkcja pipeline — zwraca najlepszą detekcję
        det = detect_flags(frame)
        if det is not None:
            draw_detection(frame, det)

        # imshow: pokazuje okno z bieżącą klatką. To szybki sposób na wizualne sprawdzenie działania.
        cv2.imshow("Flag detector (PL/RU/UA)", frame)

        # waitKey(1): potrzebne do aktualizacji okna imshow i do odczytu klawiszy.
        # Zwraca kod naciśniętego klawisza lub -1 jeśli nic nie naciśnięto.
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()