from dataclasses import dataclass
from typing import Sequence, List
from easyAI import TwoPlayerGame, Human_Player, AI_Player, Negamax

@dataclass(frozen=True)
class Rules:
    k: int = 5  # target is +K for Player 1, -K for Player 2
    moves: Sequence[int] = (1, 2, 3)  # allowed moves

class TugOfWar(TwoPlayerGame):
    """
    Arithmetic tug-of-war implemented with easyAI.

    Board is a single integer `position` constrained to [-K, +K].
    Player 1 (Plus) tries to reach +K, Player 2 (Minus) tries to reach -K.
    On its turn, a players chooses a signed move from +moves (for Player 1) or -moves (for Player 2), but moves are *directional*:
    - Player 1 can only choose from +moves (cannot exceed +K)
    - Player 2 can only choose from -moves (cannot exceed -K)
    Game ends immediately when position == +K or position == -K.
    """
    def __init__(self, players, rules: Rules = Rules(), start_position: int = 0):
        self.players = players
        self.rules = rules
        self.position = start_position # current position on the board
        self.current_player = 1 # easyAI: current players index (1 or 2)

    def possible_moves(self) -> List[int]:
        """
        1. Player 1 can only choose from +moves (cannot exceed +K)
        2. Player 2 can only choose from -moves (cannot exceed -K)
        :return: list of possible moves for the current player
        """
        k = self.rules.k

        if self.current_player == 1:
            return [+m for m in self.rules.moves if self.position + m <= k]
        else:
            return [-m for m in self.rules.moves if self.position - m >= -k]

    def make_move(self, move: int):
        self.position += move

    def unmake_move(self, move: int):
        self.position -= move

    def is_over(self) -> bool:
        return self.position == self.rules.k or self.position == -self.rules.k

    def show(self):
        print(f"[{-self.rules.k} ... {self.position} ... {self.rules.k}]")

    def scoring(self) -> int:
        """
        Return the score *from the viewpoint of Player 1 (Plus)*.
        easyAI's Negamax expects this convention.
        +inf if P1 already won, -inf if P2 already won, 0 otherwise.

        :return: score
        """

        if self.position == self.rules.k:
            return 10_000 # Player 1 (Plus) won
        elif self.position == -self.rules.k:
            return -10_000 # Player 2 (Minus) won

        return self.position



# --- Helper functions and CLI ---

def play_human_vs_ai(k: int = 5, moves: Sequence[int] = (1, 2, 3), depth: int = 8):
    """
    Play a human vs AI game.
    :param k: target position
    :param moves: allowed moves
    :param depth: AI search depth
    """
    game = TugOfWar(
        players=[Human_Player(), AI_Player(Negamax(depth=depth))],
        rules=Rules(k=k, moves=tuple(moves))
    )
    game.play()

    if game.position == k:
        print("Player 1 (Human) wins!")
    elif game.position == -k:
        print("Player 2 (AI) wins!")


def play_ai_vs_ai(k: int = 5, moves: Sequence[int] = (1, 2, 3), depth1: int = 8, depth2: int = 8):
    """
    Play an AI vs AI game.
    :param k: target position
    :param moves: allowed moves
    :param depth1: Player 1 AI search depth
    :param depth2: Player 2 AI search depth
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