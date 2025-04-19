import numpy as np

class BasicStrategyAgent():

    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3

    def __init__(self):
        
        self.hard_totals = self._create_hard_totals()
        self.soft_totals = self._create_soft_totals()
        self.pairs = self._create_pairs()
    
    def _create_hard_totals(self):
        table = np.zeros((14, 10), dtype=int)

        table[0, :] = self.HIT  # Hard 8
        table[1, 1:5] = self.DOUBLE  # Hard 9
        table[2, 0:8] = self.DOUBLE  # Hard 10
        table[3, :9] = self.DOUBLE  # Hard 11
        table[4, 2:5] = self.STAND  # Hard 12

        for i in range(5, 9):  # Hard 13-16
            table[i, 0:5] = self.STAND

        for i in range(9, 14):  # Hard 17+
            table[i, :] = self.STAND

        # return {
        #     player_total + 8: {dealer_upcard + 2: table[row, dealer_upcard]
        #                        for dealer_upcard in range(10)}
        #     for row, player_total in enumerate(range(8, 22))
        # }
        return table
    
    def _create_soft_totals(self):
        table = np.zeros((8, 10), dtype=int)

        table[0:2, 3:5] = self.DOUBLE  # A,2 and A,3
        table[2:4, 2:5] = self.DOUBLE  # A,4 and A,5
        table[4, 1:5] = self.DOUBLE    # A,6

        table[5, 0] = self.STAND       # A,7
        table[5, 1:5] = self.DOUBLE
        table[5, 5:7] = self.STAND

        table[6:, :] = self.STAND  # A,8 and A,9

        # return {
        #     soft_total + 13: {dealer_upcard + 2: table[row, dealer_upcard]
        #                       for dealer_upcard in range(10)}
        #     for row, soft_total in enumerate(range(13, 21))  # A,2 = 13 → A,10 = 21
        # }
        return table

    def _create_pairs(self):
        table = np.zeros((10, 10), dtype=int)

        # THIS IS A BOOLEAN TABLE... EITHER YES OR NO
        # IF ANSWER IS NO, WE WILL THEN REFER TO THE NEXT USEFUL TABLE
        SPLIT = 1
        table[0:2, 0:6] = SPLIT  # 2,2 and 3,3 vs 2–7
        table[2, 3:5] = SPLIT    # 4,4 vs 5–6
        # NEVER SPLIT ON 5,5
        table[4, 0:5] = SPLIT    # 6,6 vs 2–6
        table[5, 0:6] = SPLIT    # 7,7 vs 2–7
        table[6, :] = SPLIT      # 8,8 always split
        table[7, 0:5] = SPLIT    # 9,9 vs 2–6
        table[7, 6:8] = SPLIT    # 9,9 vs 8–9
        # NEVER SPLIT on 10,10
        table[9, :] = SPLIT      # A,A always split

        # return {
        #     pair_rank: {dealer_upcard + 2: table[row, dealer_upcard]
        #                 for dealer_upcard in range(10)}
        #     for row, pair_rank in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # }
        return table
    
    def _get_split_decision(self, player_total, dealer_card):
        player_card = player_total//2

        player_card_index = player_card-2
        dealer_card_index = dealer_card-2

        split_choice = self.pairs[player_card_index][dealer_card_index]

        return split_choice
    
    def _get_soft_decision(self, player_total, dealer_card, can_double):
        player_card = player_total-11 # other card that isn't an ace
    
        player_card_index = player_card-2
        dealer_card_index = dealer_card-2

        soft_choice = self.soft_totals[player_card_index][dealer_card_index]

        # if we wanted to double but couldn't
        if soft_choice == self.DOUBLE and can_double ==0:
            # stand if soft 18 or higher and can't double
            if player_card >= 7:
                return self.STAND
            # hit on soft 17 or lower if can't double
            else:
                return self.HIT

        return soft_choice
    
    def _get_hard_decision(self, player_total, dealer_card, can_double):
    
        player_card_index = player_total-8
        dealer_card_index = dealer_card-2

        hard_choice = self.hard_totals[player_card_index][dealer_card_index]

        # if we wanted to double but couldn't, we just hit instead
        if hard_choice == self.DOUBLE and can_double == 0:
            return self.HIT

        return hard_choice

    def act(self, state):
        """
        state = ("player_total": int, "dealer_card": int, "usable_ace": int, "can_double": int, "can_split": int)
        """
        player_total = int(state[0])
        dealer_card = int(state[1])
        usable_ace = int(state[2])
        can_double = int(state[3])
        can_split = int(state[4])

        if player_total == 21:
            return 1 # always stand on 21 (duh)
        # print(state)

        if can_split == 1: # if can split
            will_split = self._get_split_decision(player_total=player_total, dealer_card=dealer_card)

            # if we choose to split, we return that as our action
            if will_split == 1:
                return self.SPLIT

        if usable_ace == 1: # soft totals (we have an ace)
            return self._get_soft_decision(player_total=player_total, dealer_card=dealer_card, can_double=can_double)
    
        else: # hard totals (don't have an ace)
            # always hit on 8 or less
            if player_total <= 8:
                return self.HIT
            return self._get_hard_decision(player_total=player_total, dealer_card=dealer_card, can_double=can_double)

if __name__ == "__main__":
    agent = BasicStrategyAgent()
    print(agent.act([4, 4, 0, 1, 1]))
    
