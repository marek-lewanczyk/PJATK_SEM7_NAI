# 🎬 Movie Recommender Engine

System rekomendacji filmów i seriali oparty na danych TMDB oraz ocenach użytkowników.

## 🐍 1. Instalacja środowiska Python

### (Opcja A) venv — rekomendowane

```
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### Instalacja paczek

```bash
pip install -r requirements.txt
```

Każdy skrypt wykonuje jeden etap ETL.

## 🚀 2. Uruchamianie Recommendation Engine

```bash
python recommendation_engine.py "Marek Lewańczyk" 
```

Warunek: użytkownik musi istnieć w pliku:
```
data/group_ratings_raw.csv
```

## 📊 5. Output — interpretacja wyników

Po uruchomieniu skryptu zobaczysz:

=== TOP 5 REKOMENDACJE ===
- Breaking Bad | score=12.0 | genres=['Crime', 'Drama']
- The Witcher 3 | score=10.5 | genres=['Fantasy', 'Action']

=== TOP 5 ANTYREKOMENDACJE ===
- Fifty Shades of Grey | score=-8.0 | genres=['Romance']

Co oznacza:
	•	Wysoki score → duża zgodność z preferencjami użytkownika
	•	Niski/ujemny score → film/serial bardzo nie w stylu użytkownika

## 📁 6. Dane wejściowe

Dane wejściowe zostały przygotowane z arkusza kalulacyjnego, dla naszej grupy studenckiej - poprawione tytuły filmów.