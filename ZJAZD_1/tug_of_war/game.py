from dataclasses import dataclass
from typing import Sequence, List
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax
# Klasa przechowująca podstawowe reguły gry:
# k: graniczna wartość osi (np. od -5 d +5)
# moves: dozwolone wartości przesunięć w każdej turze
@dataclass(frozen=True)
class Rules:
    k: int = 5
    moves: Sequence[int] = (1, 2, 3)
# Główna klasa gry Tug of War
# Dziedziczy po klasie TwoPlayerGame z biblioteki easyAI
class TugOfWar(TwoPlayerGame):
    """
   Gra dwuosobowa typu „przeciąganie liny”.
    Gracz 1 (Plus) próbuje osiągnąć pozycję +K,
    a Gracz 2 (Minus) dąży do -K.
    """
    def __init__(self, players, rules: Rules = Rules(), start_position: int = 0):
        #Inicjalizacja graczy i reguł gry
        self.players = players
        self.rules = rules
        self.position = start_position # aktualna pozycja na osi liczbowej
        self.current_player = 1 #Numer aktualnego gracza (1 lub  2)

    def possible_moves(self) -> List[int]:
        """
        Zwraca liste mozliwych ruchów dla gracza.
        - Gracz 1 może wykonywać ruchy w prawo (ruchy dodatnie)
        - Gracz 2 może wykonywać ruchu w lewp (ruchy ujemne)
        """
        k = self.rules.k

        if self.current_player == 1:
            return [+m for m in self.rules.moves if self.position + m <= k]
        else:
            return [-m for m in self.rules.moves if self.position - m >= -k]

    def make_move(self, move: int):
        #Wykonuje ruch - zmnmienia pozycje o podaną wartość
        self.position += move

    def unmake_move(self, move: int):
        #Cofa wykonany ruch - używane przez alrogytm Negmax podczas analizy
        self.position -= move

    def is_over(self) -> bool:
        #Sprawdza czy pozycja osiagnęła +k lub -k - czyli czy gra się zakończyła
        return self.position == self.rules.k or self.position == -self.rules.k

    def show(self):
        #pokazuje aktualną pozycję na osi liczbowej
        print(f"[{-self.rules.k} ... {self.position} ... {self.rules.k}]")

    def scoring(self) -> int:
        """
          Zwraca ocenę bieżącego stanu gry z punktu widzenia Gracza1
        - Duża dodatnia wartość oznacza przewagę Gracza 1.
        - Duża ujemna wartość oznacza przewagę Gracza 2.
        """

        if self.position == self.rules.k:
            return 10_000 # Player 1 (Plus) won
        elif self.position == -self.rules.k:
            return -10_000 # Player 2 (Minus) won

        return self.position



# --- Funkcje pomocniczne do uruchomienia ---

def play_human_vs_ai(k: int = 5, moves: Sequence[int] = (1, 2, 3), depth: int = 8):
    """
    Uruchamia tryb: człowiek vs sztuczna inteligencja.
    - k: graniczna wartość osi
    - moves: możliwe ruchy
    - depth: głębokość analizy Negamax (im większa, tym AI "mądrzejsze")
    """
    game = TugOfWar(
        players=[Human_Player(), AI_Player(Negamax(depth=depth))],
        rules=Rules(k=k, moves=tuple(moves))
    )
    game.play()
# Po zakończeniu gry wyświetlany jest komunikat o zwycięzcy
    if game.position == k:
        print("Player 1 (Human) wins!")
    elif game.position == -k:
        print("Player 2 (AI) wins!")


def play_ai_vs_ai(k: int = 5, moves: Sequence[int] = (1, 2, 3), depth1: int = 8, depth2: int = 8):
    """
       Uruchamia tryb: AI vs AI — obie strony grają automatycznie.
       - k: graniczna wartość osi
       - moves: możliwe ruchy
       - depth1, depth2: głębokość analizy dla obu graczy
       """
    game = TugOfWar(
        players=[AI_Player(Negamax(depth=depth1)), AI_Player(Negamax(depth=depth2))],
        rules=Rules(k=k, moves=tuple(moves))
    )
    game.play()

    if game.position == k:
        print("Player 1 (AI) wins!")
    elif game.position == -k:
        print("Player 2 (AI) wins!")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Play Tug of War game.")
#     parser.add_argument("--mode", choices=["human_vs_ai", "ai_vs_ai"], default="human_vs_ai", help="Game mode")
#     parser.add_argument("--k", type=int, default=5, help="Target position (K)")
#     parser.add_argument("--moves", type=int, nargs="+", default=[1, 2, 3], help="Allowed moves")
#     parser.add_argument("--depth", type=int, default=6, help="Negamax depth (default 6)")
#     args = parser.parse_args()
#
#     moves = [int(m) for m in args.moves]
#
#     if args.mode == "human_vs_ai":
#         play_human_vs_ai(k=args.k, moves=moves, depth=args.depth)
#     else:
#         play_ai_vs_ai(k=args.k, moves=moves, depth1=args.depth, depth2=args.depth)