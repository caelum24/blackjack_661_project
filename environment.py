from deck_classes import Deck
import numpy as np

INITIAL_BANKROLL = 1000
MAX_BET = 1 # TODO -> should we get rid of the betting thing for now?
MIN_BET = 1 # TODO -> should we get rid of the betting thing for now?

class BlackjackEnv:

    hi_lo_values = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}
    zen_values = {'2': 1, '3': 1, '4': 2, '5': 2, '6': 2, '7': 1, '8': 0, '9': 0, '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': -1}
    uston_apc_values = {'2': 1, '3': 2, '4': 2, '5': 3, '6': 2, '7': 2, '8': 1, '9': -1, '10': -3, 'J': -3, 'Q': -3, 'K': -3, 'A': 0}

    def __init__(self, num_decks = 6, count_type: str = "full"):
        self.deck = Deck(num_decks=num_decks)
    
        if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc"]:
            print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc"])
            raise ValueError
        self.count_type = count_type # keep track of which card counting method is being used for training


        self.dealt_card_counts = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,
                                 '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0}
        self.initial_cards_per_value = self.deck.num_decks * 4  # 4 suits * num_decks
        self.total_cards_initial = self.deck.num_decks * 52

        # store different card counts -> Hi-Lo, Zen Count, and Uston APC
        self.hi_lo_count = 0
        self.zen_count = 0
        self.uston_apc_count = 0

        self.reset()

    def reset(self):
        # Check if we need to reshuffle before starting a new game
        if len(self.deck.cards) <= self.deck.num_decks * 52 * 0.25:
            # Reshuffle - reset the card counting as well
            self.deck.cards = []
            self.deck.create_deck()
            self.deck.shuffle()
            
            # reset card counts
            self.dealt_card_counts = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,
                                     '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0}
            self.hi_lo_count = 0
            self.zen_count = 0
            self.uston_apc_count = 0

        self.player_hand = []
        self.dealer_hand = []
        self.bankroll = INITIAL_BANKROLL
        self.current_bet = 1
        self.done = False
        self.doubled = False

        # Deal initial cards - track each card as it's dealt
        card = self.deck.deal()
        # self.dealt_card_counts[card.value] += 1
        self.update_card_counts(card.value)
        self.player_hand.append(card)

        card = self.deck.deal()
        # self.dealt_card_counts[card.value] += 1
        self.update_card_counts(card.value)
        self.dealer_hand.append(card)

        card = self.deck.deal()
        # self.dealt_card_counts[card.value] += 1
        self.update_card_counts(card.value)
        self.player_hand.append(card)

        card = self.deck.deal()
        # self.dealt_card_counts[card.value] += 1
        self.update_card_counts(card.value)
        self.dealer_hand.append(card)

        # Check for natural blackjack
        if self.calculate_hand_value(self.player_hand) == 21:
            if self.calculate_hand_value(self.dealer_hand) == 21:
                # Push
                self.done = True
                reward = 0
            else:
                # Player blackjack
                self.done = True
                reward = 1.5 * self.current_bet
                self.bankroll += reward
            return self.get_state(), reward, self.done

        return self.get_state(), 0, self.done

    def step(self, action):
        """
        Execute action and return new state, reward, and done flag
        Actions: 0 = hit, 1 = stand, 2 = double
        """
        # Can't take any action if game is over
        if self.done:
            return self.get_state(), 0, self.done

        reward = 0

        # Execute action
        if action == 0:  # Hit
            new_card = self.deck.deal()
            self.update_card_counts(new_card.value)
            self.player_hand.append(new_card)
            player_value = self.calculate_hand_value(self.player_hand)

            if player_value > 21:  # Bust
                self.done = True
                reward = -self.current_bet
            elif player_value == 21:  # Automatically stand on 21
                return self.stand()

        elif action == 1:  # Stand
            return self.stand()

        elif action == 2 and not self.doubled and len(self.player_hand) == 2:  # Double
            # Double the bet
            additional_bet = min(self.current_bet, self.bankroll)
            self.bankroll -= additional_bet
            self.current_bet += additional_bet
            self.doubled = True

            # Hit once and then stand
            new_card = self.deck.deal()
            self.update_card_counts(new_card.value)
            self.player_hand.append(new_card)
            player_value = self.calculate_hand_value(self.player_hand)

            if player_value > 21:  # Bust
                self.done = True
                reward = -self.current_bet
            else:
                return self.stand()

        return self.get_state(), reward, self.done

    def stand(self):
        # Dealer's turn
        dealer_value = self.calculate_hand_value(self.dealer_hand)

        # Dealer draws until 17 or higher
        while dealer_value < 17:
            new_card = self.deck.deal()
            self.update_card_counts(new_card.value)
            self.dealer_hand.append(new_card)
            dealer_value = self.calculate_hand_value(self.dealer_hand)

        player_value = self.calculate_hand_value(self.player_hand)

        # Determine outcome
        self.done = True
        if dealer_value > 21:  # Dealer bust
            reward = self.current_bet
            self.bankroll += 2 * self.current_bet
        elif player_value > dealer_value:  # Player wins
            reward = self.current_bet
            self.bankroll += 2 * self.current_bet
        elif player_value < dealer_value:  # Dealer wins
            reward = -self.current_bet
        else:  # Push
            self.bankroll += self.current_bet
            reward = 0

        return self.get_state(), reward, self.done

    def calculate_hand_value(self, hand):
        value = 0
        aces = 0

        for card in hand:
            card_value = card.get_numeric_value()
            if card_value == 11:
                aces += 1
            value += card_value

        # Adjust for aces if needed
        while value > 21 and aces > 0:
            value -= 10  # Change an Ace from 11 to 1
            aces -= 1

        return value

    def update_card_counts(self, value):

        # update total count
        self.dealt_card_counts[value] += 1

        # update hi_lo
        self.hi_lo_count += self.hi_lo_values[value]

        # update zen
        self.zen_count += self.zen_values[value]

        # update uston_apc
        self.uston_apc_count += self.uston_apc_values[value]

    def get_state(self):
        # get state for full count

        if self.count_type == "full":
            return self.get_full_count_state()
        
        # get state for no count with simple state
        if self.count_type == "empty":
            return self.get_simplified_state()
        
        # get state for card counting
        else:
            return self.get_system_count_state()

    def get_simplified_state(self):
        # simple state for Q learning with smaller state space
        '''
            This state getter is meant to compute the simple state space commonly used in
            q-learning for blackjack with a state lookup table to store the State-action values 
            for the agent at each state
        '''
        player_sum = self.calculate_hand_value(self.player_hand)
        dealer_up_card = self.dealer_hand[0].get_numeric_value()
        usable_ace = 1 if any(card.value == 'A' for card in self.player_hand) and player_sum <= 21 else 0
        can_double = 1 if len(self.player_hand) == 2 and not self.doubled else 0
        return np.array([player_sum, dealer_up_card, usable_ace, can_double])
    
    def get_system_count_state(self):

        '''
            This state getter gets the states if the card counting method used is one of those
            commonly used by humans. I.e. the hi_lo, zen, and uston counting methods
        '''

        player_sum = self.calculate_hand_value(self.player_hand)
        dealer_up_card = self.dealer_hand[0].get_numeric_value()
        usable_ace = 1 if any(card.value == 'A' for card in self.player_hand) and player_sum <= 21 else 0
        can_double = 1 if len(self.player_hand) == 2 and not self.doubled else 0

        # decide which counting method will go in the state
        if self.count_type == "hi_lo":
            current_count = self.hi_lo_count
        elif self.count_type == "zen":
            current_count = self.zen_count
        else: # "uston"
            current_count = self.uston_apc_count

        # could change current count, by modifying it to /remaining decks, which would help scaling out slightly
        
        return np.array([player_sum, dealer_up_card, usable_ace, can_double, current_count/self.deck.num_decks]) 

    def get_full_count_state(self):
        """
        Return the state representation for the RL agent:
        [player_sum, dealer_up_card, usable_ace, can_double, normalized_bankroll, normalized_bet, card percentages]
        """

        player_sum = self.calculate_hand_value(self.player_hand)
        dealer_up_card = self.dealer_hand[0].get_numeric_value()
        usable_ace = 1 if any(card.value == 'A' for card in self.player_hand) and player_sum <= 21 else 0
        can_double = 1 if len(self.player_hand) == 2 and not self.doubled else 0
        normalized_bankroll = self.bankroll / INITIAL_BANKROLL
        normalized_bet = self.current_bet / MAX_BET

        # Calculate number of remaining cards
        total_remaining = len(self.deck.cards)

        # Calculate card percentages based on what we know has been dealt
        card_percentages = []

        # 10 J Q K are all counted the same in blackjack
        face_cards_dealt = (self.dealt_card_counts['10'] + self.dealt_card_counts['J'] +
                            self.dealt_card_counts['Q'] + self.dealt_card_counts['K'])
        for value in ['2', '3', '4', '5', '6', '7', '8', '9']:
            remaining = self.initial_cards_per_value - self.dealt_card_counts[value]
            percentage = remaining / total_remaining if total_remaining > 0 else 0
            card_percentages.append(percentage)
        # Add combined 10-value cards
        remaining_10s = (4 * self.initial_cards_per_value) - face_cards_dealt
        percentage_10s = remaining_10s / total_remaining if total_remaining > 0 else 0
        card_percentages.append(percentage_10s)

        # Add Aces
        remaining_aces = self.initial_cards_per_value - self.dealt_card_counts['A']
        percentage_aces = remaining_aces / total_remaining if total_remaining > 0 else 0
        card_percentages.append(percentage_aces)

        # Combine basic state with card percentages
        basic_state = np.array([player_sum, dealer_up_card, usable_ace, can_double, normalized_bankroll, normalized_bet])
        full_state = np.concatenate([basic_state, np.array(card_percentages)])

        return full_state

    def place_bet(self, bet_size_factor):
        # Convert the continuous bet size factor (0-1) to an actual bet size
        bet = int(MIN_BET + bet_size_factor * (MAX_BET - MIN_BET))
        bet = max(MIN_BET, min(bet, MAX_BET, self.bankroll))

        self.current_bet = bet
        self.bankroll -= bet
        return self.get_state(), 0, self.done