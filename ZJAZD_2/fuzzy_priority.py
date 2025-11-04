"""
Fuzzy Task Prioritization – system rozmyty do ustalania priorytetów zadań
Autorzy: Marek Lewańczyk, Katarzyna Kasperek

Opis problemu (dla nietechnicznych):
-----------------------------------
W codziennym zarządzaniu backlogiem (zadaniami) potrzebujemy szybko i uczciwie
oceniać, co robić najpierw. Zamiast twardych, zero-jedynkowych reguł, stosujemy
logikę rozmytą (fuzzy logic), która pozwala łączyć „miękkie” oceny jak pilność,
wpływ, nakład pracy czy bliskość terminu i uzyskać końcowy Priorytet w skali 0–100.

Jak to działa krok po kroku:
1) Fuzzification – liczby wejściowe (np. Pilność=8) zamieniamy na stopnie
   przynależności do etykiet lingwistycznych (np. „Medium” 0.2, „High” 0.8).
2) Inference – uruchamiamy zestaw reguł IF–THEN. Każda reguła działa
   proporcjonalnie do siły spełnienia warunku (np. 0.8 aktywacji).
3) Aggregation – łączymy wpływy wszystkich reguł na wyjściu („Priority”).
4) Defuzzification – przeliczamy wynik rozmyty na konkretną liczbę 0–100.

Wejścia (min. 3, tutaj używamy 4):
- Urgency (0–10): Low, Medium, High
- Impact (0–10): Low, Medium, High
- Effort (1–13): Small, Medium, Large
- Deadline (0–30 dni): Immediate, Soon, Far

Wyjście:
- Priority (0–100): Very Low, Low, Medium, High, Critical

Reguły (przykładowy zestaw, można rozbudować):
1) IF urgency IS high AND impact IS high THEN priority IS critical
2) IF urgency IS high AND impact IS medium THEN priority IS high
3) IF impact IS high AND deadline IS immediate THEN priority IS critical
4) IF impact IS low AND effort IS large THEN priority IS low
5) IF urgency IS medium AND effort IS small THEN priority IS medium
6) IF impact IS medium AND effort IS small THEN priority IS medium
7) IF deadline IS far AND impact IS low THEN priority IS very low
8) IF urgency IS high AND deadline IS soon AND effort IS small THEN priority IS high
9) IF impact IS high AND effort IS large THEN priority IS medium

Co dostajesz po uruchomieniu programu:
- Wynik liczbowy priorytetu w konsoli (0–100) + etykieta lingwistyczna.
- Dedykowany katalog dla każdego „taska” z zapisanymi wykresami:
  • funkcje przynależności wejść i wyjścia z naniesioną wartością wejściową,
  • wykres wynikowy (agregacja i defuzyfikacja).

Wymagania:
- Python 3.9+
- numpy, matplotlib, scikit-fuzzy
  Instalacja: pip install numpy matplotlib scikit-fuzzy
"""

# ============================
# IMPORTY
# ============================
import os  # operacje na katalogach/ścieżkach
from pathlib import Path  # wygodna obsługa ścieżek
from datetime import datetime  # znacznik czasu do katalogów wynikowych
import numpy as np  # obliczenia numeryczne
import matplotlib.pyplot as plt  # rysowanie wykresów

# scikit-fuzzy – silnik logiki rozmytej
import skfuzzy as fuzz  # funkcje przynależności i defuzyfikacja
from skfuzzy import control as ctrl  # ControlSystem, Rule, Antecedent/Consequent


# ============================
# KONFIGURACJA WSZECHŚWIATÓW ZMIENNYCH (UNIVERSES)
# ============================
# Uwaga: zakresy dobrane zgodnie z opisem w zadaniu.
# Tworzymy „osie” wartości dla każdej zmiennej.
urgency_universe = np.linspace(0, 10, 101)      # 0–10, krok 0.1
impact_universe = np.linspace(0, 10, 101)       # 0–10, krok 0.1
effort_universe = np.linspace(1, 13, 121)       # 1–13, krok ~0.1
deadline_universe = np.linspace(0, 30, 301)     # 0–30 dni, krok 0.1
priority_universe = np.linspace(0, 100, 1001)   # wyjście 0–100, krok 0.1


# ============================
# DEFINICJA ZMIENNYCH ROZMYTYCH (ANTE/CONSEQUENT)
# ============================
# Antecedent – wejścia; Consequent – wyjście
urgency = ctrl.Antecedent(urgency_universe, 'urgency')
impact = ctrl.Antecedent(impact_universe, 'impact')
effort = ctrl.Antecedent(effort_universe, 'effort')
deadline = ctrl.Antecedent(deadline_universe, 'deadline')
priority = ctrl.Consequent(priority_universe, 'priority')


# ============================
# FUNKCJE PRZYNALEŻNOŚCI (MEMBERSHIP FUNCTIONS)
# ============================
# Urgency: Low (0–3), Medium (2–7), High (6–10)
urgency['low'] = fuzz.trimf(urgency_universe, [0, 0, 3])
urgency['medium'] = fuzz.trimf(urgency_universe, [2, 5, 7])
urgency['high'] = fuzz.trimf(urgency_universe, [6, 10, 10])

# Impact: Low, Medium, High – symetrycznie do Urgency
impact['low'] = fuzz.trimf(impact_universe, [0, 0, 3])
impact['medium'] = fuzz.trimf(impact_universe, [2, 5, 7])
impact['high'] = fuzz.trimf(impact_universe, [6, 10, 10])

# Effort: Small, Medium, Large (1–13). Zakładamy, że „Small” faworyzuje niskie koszty.
effort['small'] = fuzz.trimf(effort_universe, [1, 1, 5])
effort['medium'] = fuzz.trimf(effort_universe, [4, 7, 10])
effort['large'] = fuzz.trimf(effort_universe, [9, 13, 13])

# Deadline: Immediate (0–2 dni), Soon (1–7), Far (6–30)
deadline['immediate'] = fuzz.trimf(deadline_universe, [0, 0, 2])
deadline['soon'] = fuzz.trimf(deadline_universe, [1, 4, 7])
deadline['far'] = fuzz.trapmf(deadline_universe, [6, 10, 30, 30])

# Priority (wyjście): Very Low, Low, Medium, High, Critical
priority['very_low'] = fuzz.trimf(priority_universe, [0, 0, 20])
priority['low'] = fuzz.trimf(priority_universe, [10, 25, 40])
priority['medium'] = fuzz.trimf(priority_universe, [35, 50, 65])
priority['high'] = fuzz.trimf(priority_universe, [60, 75, 90])
priority['critical'] = fuzz.trimf(priority_universe, [85, 100, 100])


# ============================
# REGUŁY ROZMYTE (IF–THEN)
# ============================
rule_1 = ctrl.Rule(urgency['high'] & impact['high'], priority['critical'])
rule_2 = ctrl.Rule(urgency['high'] & impact['medium'], priority['high'])
rule_3 = ctrl.Rule(impact['high'] & deadline['immediate'], priority['critical'])
rule_4 = ctrl.Rule(impact['low'] & effort['large'], priority['low'])
rule_5 = ctrl.Rule(urgency['medium'] & effort['small'], priority['medium'])
rule_6 = ctrl.Rule(impact['medium'] & effort['small'], priority['medium'])
rule_7 = ctrl.Rule(deadline['far'] & impact['low'], priority['very_low'])
rule_8 = ctrl.Rule(urgency['high'] & deadline['soon'] & effort['small'], priority['high'])
rule_9 = ctrl.Rule(impact['high'] & effort['large'], priority['medium'])

# Zbuduj system kontrolny i symulację
priority_ctrl = ctrl.ControlSystem([
    rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9
])


# ============================
# FUNKCJE POMOCNICZE: rysowanie i ewaluacja
# ============================

def _ensure_dir(path: Path) -> None:
    """Tworzy katalog (rekurencyjnie), jeśli nie istnieje."""
    path.mkdir(parents=True, exist_ok=True)


def _plot_membership(ax, universe, variable, labels, vline=None, title=""):
    """Rysuje funkcje przynależności dla zmiennej i opcjonalnie pionową linię
    z aktualną wartością wejściową.

    :param ax: obiekt osi matplotlib
    :param universe: oś wartości (numpy array)
    :param variable: obiekt Antecedent/Consequent ze zdefiniowanymi MF
    :param labels: lista etykiet do narysowania
    :param vline: opcjonalna wartość liczbowo – pionowa linia
    :param title: tytuł wykresu
    """
    for lab in labels:  # narysuj każdą etykietę lingwistyczną
        ax.plot(universe, variable[lab].mf, label=lab)
    if vline is not None:  # pionowa linia z wartością wejściową
        ax.axvline(vline, linestyle='--')
    ax.set_title(title)
    ax.set_xlabel(variable.label)
    ax.set_ylabel('stopień przynależności')
    ax.legend(loc='best')
    ax.grid(True, linestyle=':')


def evaluate_task(urgency_val: float,
                  impact_val: float,
                  effort_val: float,
                  deadline_val: float,
                  task_name: str = "task") -> float:
    """Ewaluacja pojedynczego zadania.

    Wejście:
    - urgency_val: Pilność (0–10)
    - impact_val: Wpływ (0–10)
    - effort_val: Nakład pracy (1–13)
    - deadline_val: Dni do terminu (0–30; 0 = dziś)
    - task_name: Nazwa (użyta do wygenerowania katalogu wynikowego)

    Zasoby wyjściowe:
    - Wypis priorytetu do konsoli (0–100) + etykieta słowna.
    - Katalog results/<task_name>_<timestamp>/ z wykresami PNG.

    Zwraca:
    - priority_score: liczba 0–100
    """
    # Utwórz instancję symulatora dla tego zadania
    sim = ctrl.ControlSystemSimulation(priority_ctrl)

    # Ustaw wejścia na podstawie parametrów
    sim.input['urgency'] = float(urgency_val)
    sim.input['impact'] = float(impact_val)
    sim.input['effort'] = float(effort_val)
    sim.input['deadline'] = float(deadline_val)

    # Uruchom wnioskowanie + defuzyfikację
    sim.compute()

    # Pobierz wynik liczbowy
    priority_score = float(sim.output['priority'])

    # Ustal etykietę lingwistyczną na podstawie maksymalnej przynależności
    memberships = {
        'very_low': fuzz.interp_membership(priority_universe, priority['very_low'].mf, priority_score),
        'low': fuzz.interp_membership(priority_universe, priority['low'].mf, priority_score),
        'medium': fuzz.interp_membership(priority_universe, priority['medium'].mf, priority_score),
        'high': fuzz.interp_membership(priority_universe, priority['high'].mf, priority_score),
        'critical': fuzz.interp_membership(priority_universe, priority['critical'].mf, priority_score),
    }
    best_label = max(memberships, key=memberships.get)

    # Konsola – raport (krótko i biznesowo)
    print("=" * 60)
    print(f"Task: {task_name}")
    print(f"Wejścia: Urgency={urgency_val}, Impact={impact_val}, Effort={effort_val}, Deadline={deadline_val} dni")
    print(f"Priorytet: {priority_score:.1f} / 100  →  {best_label.upper()}")

    # Przygotuj katalog wynikowy
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in task_name)
    out_dir = Path('results') / f"{safe_name}_{ts}"
    _ensure_dir(out_dir)

    # 1) Wykresy funkcji przynależności wejść z naniesioną wartością
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    _plot_membership(axs[0, 0], urgency_universe, urgency, ['low', 'medium', 'high'], vline=urgency_val, title='Urgency – MF')
    _plot_membership(axs[0, 1], impact_universe, impact, ['low', 'medium', 'high'], vline=impact_val, title='Impact – MF')
    _plot_membership(axs[1, 0], effort_universe, effort, ['small', 'medium', 'large'], vline=effort_val, title='Effort – MF')
    _plot_membership(axs[1, 1], deadline_universe, deadline, ['immediate', 'soon', 'far'], vline=deadline_val, title='Deadline – MF')
    plt.tight_layout()
    fig.savefig(out_dir / 'inputs_membership.png', dpi=160)
    plt.close(fig)

    # 2) Wykres wyjścia – wszystkie MF + pionowa linia z wynikiem
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    _plot_membership(ax2, priority_universe, priority,
                     ['very_low', 'low', 'medium', 'high', 'critical'],
                     vline=priority_score,
                     title='Priority – wynik i MF')
    fig2.savefig(out_dir / 'priority_output.png', dpi=160)
    plt.close(fig2)

    # Zwróć wynik liczbowy
    return priority_score


# ============================
# DEMO / PRZYKŁADOWE URUCHOMIENIA
# ============================
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="System Fuzzy do priorytetyzacji zadań (autorzy: Marek Lewańczyk, Katarzyna Kasperek)"
    )
    parser.add_argument('--urgency', type=float, help='Pilność (0–10)', required=False)
    parser.add_argument('--impact', type=float, help='Wpływ (0–10)', required=False)
    parser.add_argument('--effort', type=float, help='Nakład pracy (1–13)', required=False)
    parser.add_argument('--deadline', type=float, help='Dni do terminu (0–30)', required=False)
    parser.add_argument('--name', type=str, help='Nazwa zadania (bez spacji)', default='custom_task')

    args = parser.parse_args()

    if all(v is not None for v in [args.urgency, args.impact, args.effort, args.deadline]):
        # tryb CLI – użytkownik podał argumenty
        evaluate_task(
            urgency_val=args.urgency,
            impact_val=args.impact,
            effort_val=args.effort,
            deadline_val=args.deadline,
            task_name=args.name,
        )
    else:
        # fallback: przykłady demonstracyjne
        print("Nie podano pełnych argumentów, uruchamiam przykłady demo.")
        evaluate_task(8, 7, 5, 2, 'Pilne_kluczowe_2dni')
        evaluate_task(3, 2, 8, 25, 'Niska_waznosc_duzy_wysilek')
        evaluate_task(7, 6, 2, 5, 'Quick_win')
