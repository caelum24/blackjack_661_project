from deck_classes import Deck
import numpy as np

INITIAL_BANKROLL = 1000
MAX_BET = 1 # TODO -> should we get rid of the betting thing for now?
MIN_BET = 1 # TODO -> should we get rid of the betting thing for now?

class BlackjackEnv:

    # different card counting strategies and the values associated with each card
    ten_count_values = {'2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4, '10': -9, 'J': -9, 'Q': -9, 'K': -9, 'A': 4}
    hi_lo_values = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}
    zen_values = {'2': 1, '3': 1, '4': 2, '5': 2, '6': 2, '7': 1, '8': 0, '9': 0, '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': -1}
    uston_apc_values = {'2': 1, '3': 2, '4': 2, '5': 3, '6': 2, '7': 2, '8': 1, '9': -1, '10': -3, 'J': -3, 'Q': -3, 'K': -3, 'A': 0}

    # encourage the model to try out splitting and doubling by adding early bonuses and decaying them over time, forcing the model to try them early
    # TODO -> need to implement a system to decay these rewards over time
    reward_bonuses = {"hit": 0.2, "stand":0, "double":3, "split":5} #TODO -> put this in the trainer probably
   
    
    def __init__(self, num_decks = 6, count_type: str = "full"):
        
        # initialize the deck size
        self.deck = Deck(num_decks=num_decks)
    
        # ensure the card count type is valid and set it
        if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
            print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"])
            raise ValueError
        self.count_type = count_type # keep track of which card counting method is being used for training

        # Start with an initial bankroll and a bet size (for training, self.current_bet should either be 1 or 2)
        self.bankroll = INITIAL_BANKROLL
        self.current_bet = 1

        self.dealt_card_counts = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,
                                 '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0}

        # keep track of the max amount of each type of card is in the deck
        self.initial_cards_per_value = self.deck.num_decks * 4  # 4 suits * num_decks
        self.total_cards_initial = self.deck.num_decks * 52

        # store different card counts -> Hi-Lo, Zen Count, and Uston APC
        self.hi_lo_count = 0
        self.zen_count = 0
        self.uston_apc_count = 0
        self.ten_count_count = 0

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
            self.ten_count_count = 0

        # game state tracking
        self.player_hand = []
        self.dealer_hand = []
        self.split_game_data = {"split_state":None, "ret1":None, "ret2":None}
        self.current_bet = 1
        self.done = False
        self.doubled = False
        self.split_taken = False

        # add cards to both the dealer and the player hand
        self.deal_initial_hand()

        # Check for natural blackjack
        if self.calculate_hand_value(self.player_hand) == 21:
            self.done = True
            if self.calculate_hand_value(self.dealer_hand) == 21:
                # Push
                reward = 0
            else:
                # Player blackjack
                reward = 1.5 * self.current_bet
                self.bankroll += reward
            return self.get_state(), reward, self.done

        return self.get_state(), 0, self.done

    def step(self, action, split_mode = False):
        """
        Execute action and return new state, reward, and done flag
        Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
        """
        # Can't take any action if game is over
        if self.done:
            return self.get_state(), 0, self.done

        # Execute action
        if action == 0:  # Hit
            return self.hit(split_mode = split_mode)

        elif action == 1:  # Stand
            return self.stand(split_mode = split_mode)

        elif action == 2 and not self.doubled and len(self.player_hand) == 2:  # Double
            return self.double(split_mode = split_mode)

        elif action == 3 and not self.split_taken and len(self.player_hand) == 2 and self.player_hand[0].value == self.player_hand[1].value:  #Splitting
            return self.split()
        
        else:
            print(f"SOMETHING ILLEGAL HAS HAPPENED. Action = {action}")

        # TODO -> modify this reward
        # return self.get_state(), reward, self.done

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
    
        # update ten_count
        self.ten_count_count += self.ten_count_values[value]

    def get_state(self):

        player_sum = self.calculate_hand_value(self.player_hand)
        dealer_up_card = self.dealer_hand[0].get_numeric_value()
        usable_ace = 1 if any(card.value == 'A' for card in self.player_hand) and player_sum <= 21 else 0
        can_double = 1 if len(self.player_hand) == 2 and not self.doubled else 0
        can_split = 1 if len(self.player_hand == 2) and not self.split_taken and self.player_hand[0].value == self.player_hand[1].value else 0

        basic_state = np.array([player_sum, dealer_up_card, usable_ace, can_double, can_split])

        # if we're using a simplified state, there will be nothing for counts
        counts = np.array([])

        # get state for full count
        if self.count_type == "full":
            counts = self.get_full_count_state()
        # get state for system card counting if not full count
        elif self.count_type != "empty":
            counts = self.get_system_count_state()
        
        return np.concatenate([basic_state, counts])
    
    def get_system_count_state(self):

        '''
            This state getter gets the states if the card counting method used is one of those
            commonly used by humans. I.e. the hi_lo, zen, uston, and ten_count counting methods
        '''
        # decide which counting method will go in the state
        if self.count_type == "hi_lo":
            current_count = self.hi_lo_count
        elif self.count_type == "zen":
            current_count = self.zen_count
        elif self.count_type == "uston_apc": # "uston"
            current_count = self.uston_apc_count
        else:
            current_count = self.ten_count_count

        # current count has to be scaled by the number of remaining decks, as that dictates our actual advantage
        num_remaining_decks = self.deck.get_decks_remaining() if self.deck.get_decks_remaining() > 0 else self.deck.num_decks
        return np.array([current_count/num_remaining_decks])

    def get_full_count_state(self):
        """
        Return the state representation for the RL agent:
        [player_sum, dealer_up_card, usable_ace, can_double, normalized_bankroll, normalized_bet, card percentages]
        """

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

        # card percentages is array with percentage counts for: ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
        # all face cards are rolled into 10 for this
        return np.array([card_percentages])

    def deal_initial_hand(self):
        # Deal initial cards - track each card as it's dealt
        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.player_hand.append(card)

        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.dealer_hand.append(card)

        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.player_hand.append(card)

        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.dealer_hand.append(card)

    def place_bet(self, bet_size_factor):
        # Convert the continuous bet size factor (0-1) to an actual bet size
        bet = int(MIN_BET + bet_size_factor * (MAX_BET - MIN_BET))
        bet = max(MIN_BET, min(bet, MAX_BET, self.bankroll))

        self.current_bet = bet
        self.bankroll -= bet
        return self.get_state(), 0, self.done

    def play_dealer(self):
        # play out the dealer's turn to finish the game
        # Dealer's turn
        dealer_value = self.calculate_hand_value(self.dealer_hand)

        # Dealer draws until 17 or higher
        while dealer_value < 17:
            new_card = self.deck.deal()
            self.update_card_counts(new_card.value)
            self.dealer_hand.append(new_card)
            dealer_value = self.calculate_hand_value(self.dealer_hand)
        
        return dealer_value

    def hit(self, split_mode = False):

        #deal the player a new card
        new_card = self.deck.deal()
        self.update_card_counts(new_card.value)
        self.player_hand.append(new_card)
        player_value = self.calculate_hand_value(self.player_hand)

        if split_mode and player_value >= 21:
            #if that game would end, we want to switch to the second game or be done

            # if this is in the first fork, switch to the second fork and keep going
            # TODOTODO I was working here...
            if self.split_game_data["ret1"] is None:
                self.split_game_data["ret1"] = (self.get_state(), True)
                self.switch_to_second_split()
                return self.get_state(), 0.0, self.done

        if player_value > 21:  # Bust
            self.done = True
            reward = -self.current_bet

        elif player_value == 21:  # Automatically stand on 21
            return self.stand(split_mode=split_mode)
        
        # add the reward bonus for hitting
        reward += self.reward_bonuses["hit"]

        return self.get_state(), reward, self.done

    def stand(self, hit_stop = False, double_stop = False, split_mode = False):
        
        dealer_value = self.play_dealer()
        player_value = self.calculate_hand_value(self.player_hand)

        # Determine outcome
        self.done = True

        if dealer_value > 21:  # Dealer bust
            reward = self.current_bet
        elif player_value > dealer_value:  # Player wins
            reward = self.current_bet
        elif player_value < dealer_value:  # Dealer wins
            reward = -self.current_bet
        else:  # Push
            # self.bankroll += self.current_bet
            reward = 0
        
        if double_stop:
            reward*=2
        
        # modify the bankroll based on win or loss
        self.bankroll += self.current_bet
        
        # add the standing bonus to the reward
        if hit_stop:
            # hit method uses stand as a call, so can use this to keep the reward bonus for hit
            reward += self.reward_bonuses["hit"]
        if double_stop:
            # double method uses stand as a call, so can use this to keep the reward bonus and the double bet reward for double
            reward += self.reward_bonuses["double"]
        else:
            reward += self.reward_bonuses["stand"]
        
        # switch to second split hand
        #TODO -> using done with 0.0, 0.5, 1.0
            # if done is 1.0 immediately after switching to the second split, we know we're good
        if self.split_taken:
            self.switch_to_second_split()

        return self.get_state(), reward, self.done
    
    def double(self, split_mode = False):
        # Double the bet
        additional_bet = min(self.current_bet, self.bankroll)
        self.bankroll -= additional_bet
        # self.current_bet += additional_bet
        self.doubled = True

        # Hit once and then stand
        new_card = self.deck.deal()
        self.update_card_counts(new_card.value)
        self.player_hand.append(new_card)
        player_value = self.calculate_hand_value(self.player_hand)

        if player_value > 21:  # Bust
            self.done = True
            reward = -2*self.current_bet
            return self.get_state(), reward, self.done
        else:
            return self.stand(split_mode=split_mode)
    
    def split(self):
        ## ONE SIMPLE IMPLEMENTATION

        # return self.simple_split_resolve()
        
        
        self.split_taken = True

        # add the state that caused the split to the split_game_data
        self.split_game_data["split_state"] = self.get_state()
        self.split_game_data["split_bonus"] = self.reward_bonuses["split"]
            
        # CREATE 2 SPLIT HANDS
        # split one of the cards into the split hand
        self.split_hand.append(self.player_hand.pop())

        # deal a new card to the player hand
        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.player_hand.append(card)

        # reward should usually be 0 for splitting
        reward = 0

        #TODO -> figure out what the next two
        
        # Check for natural blackjack from first hand
        #TODO -> check if blackjack off a split results in 1.5
        if self.calculate_hand_value(self.player_hand) == 21:
            blackjack_reward = 1.5 * self.current_bet
            self.bankroll += blackjack_reward
                                                # state, blackjack? -> blackjack to know if the next state is just guaranteed reward
            self.split_game_data["ret1"] = [self.get_state(), True]

            self.switch_to_second_split()
            if self.calculate_hand_value(self.player_hand) == 21:
                self.done = True
                blackjack_reward = 1.5 * self.current_bet
                self.bankroll += blackjack_reward

                self.split_game_data["ret2"] = (self.get_state(), True)

        # because we split, we don't care about training intermediate steps. We just want to train the split well
        return self.get_state(), reward, self.done
    
    def switch_to_second_split(self):
        '''
            This method is used to update the player hand to reflect the second hand of the split game sequence

            Now that we're done with the first split, we set the player hand to the next hand from the split if there was one

            NOTE -> this will mess up the get_state return, but it shouldn't matter, as the self.done means that the state-action 
            value is not considered in the final q-learning update. Additionally, we must do this so that the environment step 
            returns the next state for the model to play with, which would be our second hand's dealt state
        
        '''
        self.player_hand = self.split_hand
        self.split_hand = []

        # Deal initial cards - track each card as it's dealt
        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.player_hand.append(card)
    
    def finalize_split_game(self):

        # look at the split play data and combine it into something workable
        
        pass
    
    def simple_split_resolve(self):
        '''
            This system makes the assumption that you deal the cards for both hands simultaneously after splitting...
            That is not what actually happens, as blackjack has you play one hand and then the other, but the idea behind
            this approximtion is that whether we choose the 1st or 3rd card in the deck, both are still random, and the
            affect on the card counts is likely minimal enough that it doesn't make too large of a difference

            Then, in order to not have to play the rest of the game in 2 forks, we just terminate right there, as the data
            we get from the separate forks can also be acquired by standard gameplay for not splitting

            When trying to keep track of residual bankroll, this will lead to us not being able to accurately track bankroll stuff
        '''
        # we are terminating after this split step
        self.done = True

        # split the cards in the split hand
        self.split_hand.append(self.player_hand.pop())

        reward1 = 0
        reward2 = 0

        # deal a new card to the player hand
        card = self.deck.deal()
        self.update_card_counts(card.value)
        self.player_hand.append(card)
        if self.calculate_hand_value(self.player_hand) == 21:
            reward1 = 1.5 * self.current_bet
            


        return self.get_state(), reward, self.done



        
    def get_split_game_train_data(self):
        return self.split_game_data

    
#TODO -> need to figure out how I want to deal with splitting and hitting blackjack straight away
    # I think you just return that as the state for splitting

#TODO -> it is also possible that after the switch, the second hand dealt could be blackjack... we need to account for that too by checking

# TODO -> idea! Make the environment keep track of the states the split came from and then have the trainer ask for them at the end for training

# TODO -> change done to an integer? 1 if split done and 2 if done done?

# TODO -> make a new method to switch to the second split hand

# SPLITTING NOTES:
    # - need to play through BOTH GAMES before using the dealer to finish things off


# New Plan
    # if you split, you enter split mode
    # model stops training on intermediate steps and just keeps giving actions to states
    # environment keeps track of the results throughout
    # trainer gets that data back at the end to use for splitting training