import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

DECK_COUNT = 6

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value

    @property
    def numeric_value(self):
        if self.value in ['J', 'Q', 'K']:
            return 10
        elif self.value == 'A':
            return 11  # Ace is initially 11, will be adjusted to 1 if needed
        else:
            return int(self.value)

    def __str__(self):
        return f"{self.value} of {self.suit}"

class Deck:
    def __init__(self, num_decks=DECK_COUNT):
        self.cards = []
        self.num_decks = num_decks
        self.total_cards = self.num_decks * 52
        self.create_deck()
        self.shuffle()

    def create_deck(self):
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

        for _ in range(self.num_decks):
            for suit in suits:
                for value in values:
                    self.cards.append(Card(suit, value))

    def shuffle(self):
        random.shuffle(self.cards)

    def get_decks_remaining(self) -> float:
        # return what ratio of decks are still remaining... useful for getting the true count during a game
        return len(self.cards)/self.total_cards*self.num_decks

    def deal(self) -> Card:

        return self.cards.pop()