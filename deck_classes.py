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

    def get_numeric_value(self):
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

    def deal(self):
        if len(self.cards) <= self.num_decks * 52 * 0.25:  # Reshuffle when 75% of cards are used
            # print("Reshuffling the deck...")
            self.cards = []
            self.create_deck()
            self.shuffle()

        return self.cards.pop()