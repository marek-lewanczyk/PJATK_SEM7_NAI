"""
Arithmetic tug-of-war package (easyAI).
Exports the core game class, rules, and convenience runners.
"""

from .game import (Rules, TugOfWar, play_human_vs_ai, play_ai_vs_ai)

__all__ = [
    "Rules",
    "TugOfWar",
    "play_human_vs_ai",
    "play_ai_vs_ai",
]