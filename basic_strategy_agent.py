import numpy as np

import numpy as np

class BasicStrategyAgent:
    def __init__(self):
        self.hard_totals, self.soft_totals, self.pairs = self._create_basic_strategy_table()

    def act(self, state, valid_actions=None, e_greedy=False):
        player_sum    = int(state[0])
        dealer_upcard = int(state[1])
        usable_ace    = int(state[2])
        can_double    = int(state[3])
        
        dealer_idx = min(dealer_upcard - 2, 9)

        if player_sum % 2 == 0 and 2 <= (player_sum // 2) <= 11:
            card = player_sum // 2
            row  = card - 2
            action = int(self.pairs[row][dealer_idx])
            if action == 3:
                if card == 11:
                    action = 0
                elif card == 5 and can_double:
                    action = 2
                else:
                    action = 1 if player_sum >= 17 else 0

        elif usable_ace and 13 <= player_sum <= 21:
            row = player_sum - 13
            action = int(self.soft_totals[row][dealer_idx])

        else:
            row = player_sum - 8
            action = int(self.hard_totals[row][dealer_idx]) if 0 <= row < 18 else 0

        if action == 2 and not can_double:
            action = 0

        if valid_actions is not None and action not in valid_actions:
            action = 1 if 1 in valid_actions else 0

        return action


    def _create_basic_strategy_table(self):
        """Create blackjack basic strategy tables"""
        # Hard totals basic strategy (rows=player total, cols=dealer upcard)
        # 0=Hit, 1=Stand, 2=Double
        hard_totals = np.zeros((18, 10), dtype=int)

        # Fill in the basic strategy for hard totals (8-17 vs 2-Ace)
        # Row 0 = hard 8, Row 1 = hard 9, ..., Row 9 = hard 17
        # Col 0 = dealer 2, Col 1 = dealer 3, ..., Col 9 = dealer Ace

        # Always hit for hard 8 or less
        hard_totals[0, :] = 0  # Hard 8 - always hit

        # Hard 9
        hard_totals[1, 2:6] = 2  # Double vs 3-6

        # Hard 10
        hard_totals[2, 0:9] = 2  # Double vs 2-10

        # Hard 11
        hard_totals[3, :] = 2  # Double vs everything

        # Hard 12
        hard_totals[4, 3:6] = 1  # Stand vs 4-6

        # Hard 13-16
        for i in range(5, 9):
            hard_totals[i, 0:6] = 1  # Stand vs 2-6

        # Hard 17+
        for i in range(9, 18):
            hard_totals[i, :] = 1  # Always stand

        # Soft totals basic strategy (A,2 through A,10)
        # 0=Hit, 1=Stand, 2=Double
        soft_totals = np.zeros((9, 10), dtype=int)

        # A,2 and A,3 (13-14)
        soft_totals[0:2, 4:6] = 2  # Double vs 5-6

        # A,4 and A,5 (15-16)
        soft_totals[2:4, 3:6] = 2  # Double vs 4-6

        # A,6 (17)
        soft_totals[4, 2:6] = 2  # Double vs 3-6

        # A,7 (18)
        soft_totals[5, 2:6] = 2  # Double vs 3-6
        soft_totals[5, 0:8] = 1  # Stand vs 2-9

        # A,8 and A,9 (19-20)
        soft_totals[6:8, :] = 1  # Always stand

        # Pairs basic strategy
        # 0=Hit, 1=Stand, 2=Double, 3=Split
        pairs = np.zeros((10, 10), dtype=int)

        # 2,2 and 3,3
        pairs[0:2, 2:7] = 3  # Split vs 3-7

        # 4,4
        pairs[2, 4:6] = 3  # Split vs 5-6

        # 5,5 - treat as hard 10
        pairs[3, 0:9] = 2  # Double vs 2-10

        # 6,6
        pairs[4, 2:6] = 3  # Split vs 3-6

        # 7,7
        pairs[5, 0:7] = 3  # Split vs 2-7

        # 8,8
        pairs[6, :] = 3  # Always split

        # 9,9
        pairs[7, 0:6] = 3  # Split vs 2-6
        pairs[7, 8] = 3    # Split vs 9

        # 10,10
        pairs[8, :] = 1  # Always stand

        # A,A
        pairs[9, :] = 3  # Always split

        return hard_totals, soft_totals, pairs