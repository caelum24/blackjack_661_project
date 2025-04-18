from deck_classes import Deck
import numpy as np
from hyperparameters import HyperParameters

#TODO -> make a hyperparameter for how low deck needs to get before shuffling on reset
#TODO -> make self.player_bet an input to the reset function so we can simulate our playing agent with our betting agent
#TODO may want to change the rewrd thing from _next_hand to None or remove it, as you only change hands when done, and 
#       we don't know the reward yet for a split until after all hands have gone
#TODO -> need to implement reward structure bonus

# TODO -> Card counts off by 1 or 2 sometimes, but unsure as to why... could be due to temporal differences with dealer card facing down
class BlackjackEnv:

    # different card counting strategies and the values associated with each card
    ten_count_values = {'2': 4, '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4, '10': -9, 'J': -9, 'Q': -9, 'K': -9, 'A': 4}
    hi_lo_values = {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1}
    zen_values = {'2': 1, '3': 1, '4': 2, '5': 2, '6': 2, '7': 1, '8': 0, '9': 0, '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': -1}
    uston_apc_values = {'2': 1, '3': 2, '4': 2, '5': 3, '6': 2, '7': 2, '8': 1, '9': -1, '10': -3, 'J': -3, 'Q': -3, 'K': -3, 'A': 0}

    def __init__(self, num_decks = 6, count_type: str = "full"):
        
        ### GAME STATE INITIALIZATION ###
        self.deck = Deck(num_decks=num_decks)

        ### BANKROLL STUFF ###
        self.bankroll = HyperParameters.INITIAL_BANKROLL
        self.player_bet = 1
    
        ### CARD COUNTING STUFF ###
        # ensure the card count type is valid and set it
        if count_type not in ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"]:
            print("count type must be one of", ["full", "empty", "hi_lo", "zen", "uston_apc", "ten_count"])
            raise ValueError

        # keep track of which card counting method is being used for training
        self.count_type = count_type 

        # keep track of what cards have been dealt
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
            self.deck.create_deck()
            self.deck.shuffle()
            
            # reset card counts
            self.dealt_card_counts = {'2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0,
                                     '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0}
            self.hi_lo_count = 0
            self.zen_count = 0
            self.uston_apc_count = 0
            self.ten_count_count = 0


        ### GAME STATE SETUP ###
        ### ENV keeps track of the hands, bet, and rewards. Agent must keep track of terminal states for split learning
        self.player_hands = [[]]
        self.player_bet = 1 
        self.player_reward_magnitudes = [1]
        self.rewards = []
        self.dealer_hand = []
        self.current_hand_index = 0
        self.done = 0

        self._deal_initial_hand()

        reward = 0 # reward doesn't matter here because it's not being used to teach a previous state
        # Check for natural blackjack
        if self._calculate_hand_value(self.player_hands[self.current_hand_index]) == 21:
            self.done = 2
            if self._calculate_hand_value(self.dealer_hand) == 21:
                # Push
                reward = 0
            else:
                # Player blackjack
                reward = 1.5
                # if the player has blackjack, we want to update the reward magnitudes
                self.player_reward_magnitudes[self.current_hand_index] = reward
            
            # make sure to update the card counts for the dealer hand you never saw
            self._update_card_counts(self.dealer_hand[0].value)
            


            # got some blackjack at the beginning... done = 2
            return self._get_state(), reward, 2

        return self._get_state(), reward, 0


    def step(self, action):
        """
        Execute action and return new state, reward, and done flag
        Actions: 0 = hit, 1 = stand, 2 = double, 3 = split
        """

        # Can't take any action if game is over
        if self.done:
            return self._get_state(), 0, self.done
        
        # player_hand = self.player_hands[self.current_hand_index]

        # Execute action
        if action == 0:  # Hit
            return self._hit()

        elif action == 1:  # Stand
            return self._stand()

        elif action == 2 and self._can_double():  # Double
            return self._double()

        elif action == 3 and self._can_split():  #Splitting
            return self._split()
        
        else:
            raise ValueError(f"SOMETHING ILLEGAL HAS HAPPENED. Action = {action}")

    def get_count(self):
        if self.count_type == "full":
            return self._get_full_count_state()

        elif self.count_type == "empty":
            return np.array([])
    
        else: # "hi_lo", "zen", "uston_apc", "ten_count"
            return self._get_system_count_state()
        
    def _hit(self):

        player_hand = self.player_hands[self.current_hand_index]

        #deal the player a new card and count it
        new_card = self.deck.deal()
        self._update_card_counts(new_card.value)
        player_hand.append(new_card)
        hand_value = self._calculate_hand_value(player_hand)

        # if bust, we note the reward and move on to the next hand
        if hand_value > 21:
            reward = -1 # TODO -> this reward doesn't really do anything
            # self.player_reward_magnitudes[self.current_hand_index] = reward  # note bust for the ith hand
            return self._next_hand(reward)
        
        # not done, so pass 0 for done
        return self._get_state(), 0, 0 

    def _stand(self):
        reward = 0
        return self._next_hand(reward) 
    
    def _double(self):
        player_hand = self.player_hands[self.current_hand_index]

        #deal the player a new card and count it
        new_card = self.deck.deal()
        self._update_card_counts(new_card.value)
        player_hand.append(new_card)
        hand_value = self._calculate_hand_value(player_hand)

        self.player_reward_magnitudes[self.current_hand_index] *= 2

        # if bust, we note the reward and move on to the next hand
        if hand_value > 21:
            reward = -2
            return self._next_hand(reward)

        reward=0
        return self._next_hand(reward)
        
    def _split(self):

        hand = self.player_hands[self.current_hand_index]

        card1, card2 = hand

        # deal a new card to the player hand
        new_card1 = self.deck.deal()
        self._update_card_counts(new_card1.value)
        # new_card2 = self.deck.deal()
        # self._update_card_counts(new_card2.value)

        new_hand1 = [card1, new_card1]
        # new_hand2 = [card2, new_card2]
        new_hand2 = [card2]

        self.player_hands[self.current_hand_index] = new_hand1
        # insert a hand (in case we do recursive splitting, you recurse down before going to next hand)
        self.player_hands.insert(self.current_hand_index + 1, new_hand2)
        self.player_reward_magnitudes.insert(self.current_hand_index + 1, self.player_reward_magnitudes[self.current_hand_index])

        reward = 0
        # thankfully, splitting doesn't result in blackjack, so we will never be done upon a split
        return self._get_state(), reward, 0

    def _deal_initial_hand(self):
        # Deal initial cards - track each card as it's dealt
        player_hand = self.player_hands[0]
        
        card = self.deck.deal()
        self._update_card_counts(card.value)
        player_hand.append(card)

        # dealer down card
        card = self.deck.deal()
        # self._update_card_counts(card.value) # Dealer second card is unknown until the end of the game
        self.dealer_hand.append(card)

        card = self.deck.deal()
        self._update_card_counts(card.value)
        player_hand.append(card)

        # dealer up card
        card = self.deck.deal()
        self._update_card_counts(card.value) 
        self.dealer_hand.append(card)
    
    def _update_card_counts(self, value):

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

    def _next_hand(self, reward):
        
        if self.current_hand_index+1 >= len(self.player_hands):
            return self._resolve_game()
        else:
            self.current_hand_index += 1
            
            # deal 2nd card to player's new split hand
            player_hand = self.player_hands[self.current_hand_index]
            new_card2 = self.deck.deal()
            self._update_card_counts(new_card2.value)
            player_hand.append(new_card2)
            # return done = 1 to show that the hand is over but the game isn't
            # Important to NOTE that this reward is useless, as we don't know the dealer's result yet
            return self._get_state(), reward, 1

    def _calculate_hand_value(self, hand):
        value = 0
        aces = 0

        for card in hand:
            card_value = card.numeric_value
            if card_value == 11:
                aces += 1
            value += card_value

        # Adjust for aces if needed
        while value > 21 and aces > 0:
            value -= 10  # Change an Ace from 11 to 1
            aces -= 1

        return value

    def _can_split(self):
        player_hand = self.player_hands[self.current_hand_index]
        return 1 if len(player_hand) == 2 and player_hand[0].numeric_value == player_hand[1].numeric_value and len(self.player_hands)==1 else 0 # ensure you can only split once
        # can_split = 1 if len(player_hand == 2) and player_hand[0].value == player_hand[1].value else 0
    
    def _usable_ace(self, hand):
        value = 0
        aces = 0

        for card in hand:
            card_value = card.numeric_value
            if card_value == 11:
                aces += 1
            value += card_value

        # Adjust for aces if needed
        while value > 21 and aces > 0:
            value -= 10  # Change an Ace from 11 to 1
            aces -= 1

        return 1 if aces > 0 else 0

    def _can_double(self):
        return 1 if len(self.player_hands[self.current_hand_index]) == 2 else 0

    def _get_state(self):
        player_hand = self.player_hands[self.current_hand_index]

        player_sum = self._calculate_hand_value(player_hand)
        dealer_up_card = self.dealer_hand[1].numeric_value
        usable_ace = self._usable_ace(player_hand)
        can_double = self._can_double()
        can_split = self._can_split()
        

        basic_state = np.array([player_sum, dealer_up_card, usable_ace, can_double, can_split])

        # if we're using a simplified state, there will be nothing for counts
        counts = np.array([])

        # get state for full count
        if self.count_type == "full":
            counts = self._get_full_count_state()
        # get state for system card counting if not full count
        elif self.count_type != "empty":
            counts = self._get_system_count_state()
        return np.concatenate([basic_state, counts])
    
    def _get_system_count_state(self):

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

    def _get_full_count_state(self):
        """
        Return the state representation for the RL agent:
        [player_sum, dealer_up_card, usable_ace, can_double, normalized_bankroll, normalized_bet, card percentages]
        """

        # Calculate number of remaining cards
        total_remaining = len(self.deck.cards)
        # print(total_remaining)
        # Calculate card percentages based on what we know has been dealt
        card_percentages = []

        # 10 J Q K are all counted the same in blackjack
        face_cards_dealt = (self.dealt_card_counts['10'] + self.dealt_card_counts['J'] +
                            self.dealt_card_counts['Q'] + self.dealt_card_counts['K'])

        for value in ['2', '3', '4', '5', '6', '7', '8', '9']:
            remaining = self.initial_cards_per_value - self.dealt_card_counts[value]
            # print(remaining)
            percentage = remaining / total_remaining if total_remaining > 0 else 0
            card_percentages.append(percentage)

        # Add combined 10-value cards
        remaining_10s = (4 * self.initial_cards_per_value) - face_cards_dealt
        percentage_10s = remaining_10s / total_remaining if total_remaining > 0 else 0
        # print(remaining_10s)
        card_percentages.append(percentage_10s)

        # Add Aces
        remaining_aces = self.initial_cards_per_value - self.dealt_card_counts['A']
        percentage_aces = remaining_aces / total_remaining if total_remaining > 0 else 0
        # print(remaining_aces)
        card_percentages.append(percentage_aces)

        # print(total_remaining, (self.initial_cards_per_value * 13 - sum(self.dealt_card_counts.values())))
        # if total_remaining != (self.initial_cards_per_value * 13 - sum(self.dealt_card_counts.values())):
            # print(card_percentages)
        # card percentages is array with percentage counts for: ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']
        # all face cards are rolled into 10 for this
        return np.array(card_percentages)

    def _resolve_game(self): 
        
        # finish the dealer hand
        while self._calculate_hand_value(self.dealer_hand) < 17:
            new_dealer_card = self.deck.deal()
            self._update_card_counts(new_dealer_card.value)
            self.dealer_hand.append(new_dealer_card)
             
        dealer_val =  self._calculate_hand_value(self.dealer_hand)

        # IN THIS PART, WE NEED TO LOOK AT EVERY MAGNITUDE TO GET THE BANKROLL BASED ON THE BETS AND WINS/LOSSES
        # total_reward = 0

        # compute the rewards for each hand of the game (and correspondingly each terminal state)
        self.rewards = []
        for i, hand in enumerate(self.player_hands):
            reward_magnitude = self.player_reward_magnitudes[i]

            if reward_magnitude == 1.5:
                self.bankroll += (reward_magnitude * self.player_bet)
                self.rewards.append(reward_magnitude)
                continue
            
            val = self._calculate_hand_value(hand)
            reward = reward_magnitude
            if val > 21:
                reward = -1*reward_magnitude
            elif dealer_val > 21 or val > dealer_val:
                reward = 1*reward_magnitude
            elif val == dealer_val:
                reward = 0*reward_magnitude
            else:
                reward = -1*reward_magnitude
            
            self.bankroll += (reward * self.player_bet)
            self.rewards.append(reward)
            # total_reward += reward

        # finally, update the card counter to reflect the dealer's bottom card at the beginning
        self._update_card_counts(self.dealer_hand[0].value)

        # print statements for testing purposes
        # print([self._calculate_hand_value(x) for x in self.player_hands])
        # print(dealer_val)
        
        self.done = 2
        
        # send back the final reward... if there was no split, this will be normal and learned from
        return self._get_state(), reward, 2

    def deliver_rewards(self):
        return self.rewards

    def decks_remaining(self):
        return len(self.deck.cards) / 52.0

if __name__ == "__main__":
    env = BlackjackEnv(count_type="full")
    done = False
    for i in range(1000):
        if done:
            done = False
            env._get_full_count_state()
            state, _, done = env.reset()
        else:
            state, _, done = env.step(1)
        # print(state)
        # env._get_full_count_state()



    