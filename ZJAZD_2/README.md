# README – Fuzzy Task Prioritization System

**Autorzy:** Marek Lewańczyk, Katarzyna Kasperek  
**Język:** Python 3.9+  
**Biblioteki:** numpy, matplotlib, scikit-fuzzy

---

## 🎯 Cel projektu
Ten projekt prezentuje **system logiki rozmytej (fuzzy logic)**, który pomaga automatycznie ustalać priorytety zadań projektowych.

Zamiast twardych reguł („zrób to pierwsze, bo jest pilne”), system ocenia **pilność, wpływ, wysiłek i czas do terminu**, a następnie generuje końcowy **priorytet (0–100)** wraz z etykietą lingwistyczną: Very Low, Low, Medium, High, Critical.

---

## ⚙️ Instalacja środowiska

### 1️⃣ Klonowanie lub pobranie projektu
```bash
git clone https://github.com/<twoje_repo>/fuzzy-priority.git
cd fuzzy-priority
```

Lub po prostu pobierz plik `.py` i uruchom w dowolnym katalogu.

### 2️⃣ Utworzenie wirtualnego środowiska (zalecane)
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Instalacja zależności
```bash
pip install numpy scipy matplotlib scikit-fuzzy networkx
```

---

## ▶️ Uruchomienie programu

Uruchom skrypt demonstracyjny w terminalu:
```bash
python fuzzy_priority.py
```

Uruchom skrypt z własnymi wartościami:
```bash
python fuzzy_priority.py --urgency 8 --impact 7 --effort 5 --deadline 2 --task_name "Pilne_kluczowe_2dni"
```

Wynik pojawi się w konsoli w formacie:
```
============================================================
Task: Pilne_kluczowe_2dni
Wejścia: Urgency=8, Impact=7, Effort=5, Deadline=2 dni
Priorytet: 85.4 / 100  →  CRITICAL
```

Dodatkowo utworzony zostanie katalog:
```
results/Pilne_kluczowe_2dni_20251105_142201/
```
Zawartość:
- `inputs_membership.png` – wykresy przynależności wejść (Urgency, Impact, Effort, Deadline)
- `priority_output.png` – wynikowy priorytet z naniesioną linią wyniku

---

## 🧠 Zmienne wejściowe
| Nazwa | Znaczenie | Zakres | Etykiety rozmyte |
|--------|------------|--------|-------------------|
| **Urgency** | Jak pilne jest zadanie | 0–10 | Low, Medium, High |
| **Impact** | Jak duży wpływ ma zadanie na projekt | 0–10 | Low, Medium, High |
| **Effort** | Ile wysiłku wymaga zadanie | 1–13 | Small, Medium, Large |
| **Deadline** | Ile dni pozostało do końca | 0–30 | Immediate, Soon, Far |

---

## 📈 Wyjście systemu
| Nazwa | Znaczenie | Zakres | Etykiety rozmyte |
|--------|------------|--------|-------------------|
| **Priority** | Końcowy priorytet zadania | 0–100 | Very Low, Low, Medium, High, Critical |

---

## 🧩 Zasady rozmyte (IF–THEN)
Przykładowe reguły:
```
1. IF urgency IS high AND impact IS high THEN priority IS critical
2. IF urgency IS high AND impact IS medium THEN priority IS high
3. IF impact IS high AND deadline IS immediate THEN priority IS critical
4. IF impact IS low AND effort IS large THEN priority IS low
5. IF urgency IS medium AND effort IS small THEN priority IS medium
6. IF impact IS medium AND effort IS small THEN priority IS medium
7. IF deadline IS far AND impact IS low THEN priority IS very low
8. IF urgency IS high AND deadline IS soon AND effort IS small THEN priority IS high
9. IF impact IS high AND effort IS large THEN priority IS medium
```

---

## 🧪 Przykłady użycia
```python
# Pilne i ważne – termin 2 dni
evaluate_task(urgency_val=8, impact_val=7, effort_val=5, deadline_val=2, task_name='Pilne_kluczowe_2dni')

# Mało ważne, dużo pracy – termin odległy
evaluate_task(urgency_val=3, impact_val=2, effort_val=8, deadline_val=25, task_name='Niska_waznosc_duzy_wysilek')

# Szybkie zwycięstwo
evaluate_task(urgency_val=7, impact_val=6, effort_val=2, deadline_val=5, task_name='Quick_win')
```

---

## 📊 Wizualizacja wyników
Każdy task tworzy osobny folder z wykresami:
- **Inputs:** funkcje przynależności i linie pionowe z wartością wejściową.
- **Priority Output:** wynik rozmyty z zaznaczoną wartością końcową.

Wizualizacje są zapisywane w formacie `.png` i można je wykorzystać w raportach lub prezentacjach.

---

## 🏁 Podsumowanie
Projekt demonstruje w praktyce zastosowanie logiki rozmytej w zarządzaniu zadaniami.
Pozwala podejmować decyzje w sposób bardziej elastyczny i zbliżony do ludzkiego myślenia.