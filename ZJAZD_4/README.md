# ZJAZD_4 - Instrukcja uruchomienia

Ten katalog zawiera skrypt `main.py`, który uruchamia prosty pipeline uczenia maszynowego (preprocessing, PCA, DecisionTree, SVM) i generuje wykresy oraz raporty.

## Wymagania
- Python 3.10+ (skrypt testowany na Python 3.10–3.13). 
- pip

Zależności Python znajdują się w pliku `requirements.txt` w katalogu.

## Szybkie uruchomienie (zalecane: wirtualne środowisko)
1. Utwórz i aktywuj wirtualne środowisko (zsh/macOS):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Zainstaluj zależności (jeżeli uruchamiasz z katalogu `ZJAZD_4`, użyj ścieżki do pliku w katalogu wyżej):

```bash
pip install -r requirements.txt
```

3. Uruchom skrypt:

```bash
python3 main.py
```

Skrypt wypisze raporty do konsoli oraz zapisze wykresy (np. `histogram_labels.png`, `scatter_plot.png`, `pca2d.png`) w bieżącym katalogu.

## Zmiana zestawu danych (plik wejściowy)
- Domyślnie `main.py` oczekuje pliku `data/titanic.csv` (ścieżka ustawiona w kodzie jako `FILE_PATH = "data/titanic.csv"`).
- Aby użyć innego pliku CSV/TXT:
  - Najprościej: podmień plik w katalogu `ZJAZD_4/data/` i nazwij go `titanic.csv`, lub
  - Edytuj zmienną `FILE_PATH` w `main.py` i ustaw ścieżkę/URL do twojego pliku danych, np. `FILE_PATH = "data/my_dataset.csv"`.

Uwaga do formatu danych:
- Skrypt próbuje automatycznie wykryć separator (używa `pd.read_csv(..., sep=None, engine='python')`), ale jeśli wczytanie się nie powiedzie, spróbuje standardowego `pd.read_csv(FILE_PATH)`.
- Skrypt automatycznie detektuje kolumnę celu (target): jeżeli znajdzie kolumnę nazwaną `target`, `label`, `class`, `y` lub `Y` - użyje jej; w przeciwnym razie użyje ostatniej kolumny jako target.
- Jeśli target jest numeryczny i ma więcej niż 10 unikalnych wartości, skrypt zadziała jako klasyfikator binarny (dzieli według mediany).
- Jeżeli chcesz wskazać konkretną kolumnę celu bez edycji kodu, po prostu nazwij ją `target` lub `label` w pliku wejściowym.

## Co robi skrypt – krótkie podsumowanie
- Wczytuje plik CSV/TXT
- Zamienia przecinki na kropki w wartościach tekstowych (przydatne przy europejskim formacie liczb)
- Próbuje skonwertować kolumny na typy numeryczne tam, gdzie jest to możliwe
- Usuwa wiersze z brakami (synchronizuje X i y)
- Preprocessing: standaryzacja kolumn numerycznych + OneHotEncoding kolumn kategorycznych
- Redukcja wymiaru: PCA 2D (jeśli jest >1 cecha) i zapis wykresu `pca2d.png`
- Trening modeli: DecisionTree (max_depth=5) i SVM (różne kernele) - metryki wypisywane są w konsoli
- Zapis prostych wykresów: histogram lub scatter oraz PCA

## Wyjścia (pliki wygenerowane)
- `histogram_labels.png` - histogram rozkładu klas (jeśli brak kolumn X/Y do scatter)
- `scatter_plot.png` - scatter plot (jeśli obecne kolumny `X` i `Y`)
- `pca2d.png` - wizualizacja PCA 2D (jeśli >=2 cechy)
- Konsola - szczegółowe raporty (`classification_report`, macierz pomyłek) dla DecisionTree i metryki dla SVM

## Uruchomienie kodu dla zestawu danych Swedish Auto Insurance:

Wzięliśmy dane, w których:
	•	X oznacza jakąś liczbę związaną z ubezpieczeniami (np. liczbę polis, zgłoszeń)
	•	Y to koszt, który chcemy przewidywać

Ponieważ Y było liczbą, podzieliliśmy dane na dwie grupy:
	•	0 - niższy koszt
	•	1 - wyższy koszt

### Na wykresie widać coś takiego:
Im większe X, tym większy koszt Y.
Czyli: większa liczba X oznacza większe koszty.

Dane są bardzo uporządkowane i tworzą prawie prostą linię.
To oznacza, że modele mają łatwe zadanie.

### Podsumowanie działania kernel functions w SVM
W projekcie przetestowano cztery różne metody pracy modelu SVM (kernels): linear, rbf, poly oraz sigmoid. Dane są prawie liniowe, dlatego kernel linear osiąga najlepsze wyniki (~74% accuracy). Kernele rbf i sigmoid nie poprawiają jakości - dają identyczne wyniki co linear, ponieważ dane nie mają zakrzywionych struktur. Kernel poly wypada najsłabiej, ponieważ próbuje tworzyć zbyt skomplikowaną granicę między klasami, co pogarsza jakość klasyfikacji. Najprostsze rozwiązanie (kernel linear) działa najlepiej.

### Użyliśmy dwóch rodzajów modeli:

1) Drzewo decyzyjne
	- działa w stylu „jeśli X jest większe niż…, to koszt będzie wyższy”
	- wypadło średnio, poprawnie rozpoznaje tańsze przypadki, ale gorzej te droższe
	- dokładność: ok. 68%

2) SVM (bardziej „matematyczny” model)
	- tutaj wyniki były lepsze
	- model dawał ok. 74% poprawnych przewidywań
	- najlepsze okazały się metody „liniowe”, bo dane są prawie jak linia na wykresie

### Przykład przewidywania
Dla X = 7 model mówi - koszt będzie niski (klasa 0)
I to zgadza się z wykresem - małe X, mały koszt.

### Najprostszy możliwy wniosek
Większa wartość X prawie zawsze oznacza wyższy koszt Y. Najlepszy jest model liniowy (SVM), który przewiduje koszty z dokładnością ok. 74%.

## Uruchomienie kodu dla zestawu danych Titanic:

Wzięliśmy dane z Titanica, w których chcemy przewidywać skąd pasażer wsiadł na statek.

Możliwe wartości:
	•	S – Southampton
	•	C – Cherbourg
	•	Q – Queenstown
	•	nan – brak informacji

To oznacza, że jest to zadanie wieloklasowe (4 klasy).

### Na wykresie widać coś takiego:

1) Histogram klas

Histogram pokazuje wyraźnie:
- prawie 600 osób wsiadło w Southampton (S) → ogromna większość
- znacznie mniej w Cherbourg (C)
- jeszcze mniej w Queenstown (Q)
- jedna osoba z brakującą informacją (nan)

Oznacza to, że dane są bardzo niezbalansowane - jedna klasa dominuje wszystkie pozostałe.

2) PCA

PCA to rzutowanie danych na płaszczyznę (2D), aby zobaczyć ich układ.**

Tutaj punkty różnych klas są:
- mieszane ze sobą
- nie tworzą wyraźnych grup
- nie da się ich oddzielić w prosty sposób

To znaczy, że modelom będzie trudno przewidzieć klasę „Embarked”, bo cechy nie rozdzielają tych klas w naturalny sposób.

### Użyliśmy dwóch rodzajów modeli:

1) Drzewo decyzyjne

Drzewo działa jak seria pytań typu:
„Jeśli wiek większy niż…”, „jeśli cena biletu mniejsza niż…”, itd.

Wynik:
- świetnie rozpoznaje klasę S (bo jej jest najwięcej)
- radzi sobie gorzej z klasami C i Q (bo mało danych)
- klasę nan kompletnie pomija (bo jest tylko 1 przypadek)
- doładność ~ 80%, ale głównie dzięki temu, że klasa S jest tak częsta

W praktyce:
Model „przechyla się” w stronę najczęstszej klasy.

2) SVM (różne wersje: linear, rbf, poly, sigmoid)

SVM próbuje narysować granice oddzielające klasy.

Wyniki:
	•	linear → accuracy ok. 87%
	•	rbf → ok. 79%
	•	poly → ok. 80%
	•	sigmoid → ok. 76%

Co z tego wynika?
	•	linear wypada najlepiej, bo dla tej liczby cech prosta granica często „wystarczy”
	•	rbf i sigmoid próbują wyginać granice, co nie pomaga - dane nie mają sensownej struktury
	•	poly (wielomian) próbuje kombinować, ale efekt końcowy wciąż jest gorszy niż linear

Największy problem to nierównowaga klas, przez co:
	•	metryki dla rzadkich klas (C, Q) są słabe,
	•	model może przewidywać głównie S i i tak osiągać wysoką accuracy.

### Przykład przewidywania
Dla przykładowego pasażera:
Model przewidział, że wsiadł w Cherbourg (C).
To oznacza, że jego cechy (bilet, płeć, wiek, cena biletu, klasa) bardziej pasują do osób wsiadających w C niż w S czy Q.

### Najprostszy możliwy wniosek

1. Najwięcej pasażerów wsiadło z Southampton (S) - dlatego modele chętnie przewidują tę klasę.
2.	Dane są bardzo nierównomierne, dlatego trudno nauczyć model odróżniać rzadkie klasy.
3.	Model liniowy SVM radzi sobie najlepiej (ok. 87%), ale głównie dlatego, że elegancko „wyłapuje” dominującą klasę.
4.	PCA pokazuje totalne wymieszanie danych, co oznacza, że cechy nie mówią jasno, w którym porcie ktoś wsiadł.
5.	Drzewo działa nieźle, ale też głównie dzięki przewidywaniu klasy S.

---
Autorzy: Marek Lewańczyk s29420, Katarzyna Kasperek s27553

