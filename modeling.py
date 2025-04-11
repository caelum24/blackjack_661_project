import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
from environment import BlackjackEnv 


def model(agent):
    '''
    Wrapper to create the strategy table and compare agent results
    Pass the trained agent and this module will print the strategy comparisons
    '''
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_basic_strategy_table():
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

    def compare_with_basic_strategy(agent):
        """Compare agent decisions with basic strategy for all combinations"""
        hard_totals, soft_totals, pairs = create_basic_strategy_table()

        # Create mapping for action names
        action_names = {0: "Hit", 1: "Stand", 2: "Double"}

        # Prepare the results
        print("\n\n===== COMPARING AGENT VS BASIC STRATEGY =====\n")

        # Create a figure for the comparison visualization
        plt.figure(figsize=(18, 15))

        # 1. HARD TOTALS
        print("\nHARD TOTALS COMPARISON:")
        print("Player Total | Dealer Upcard | Basic Strategy | Agent Decision | Match?")
        print("-" * 75)

        # Prepare matrix for visualization
        comparison_matrix = np.zeros((18, 10))

        # Check each combination
        for player_total in range(8, 26):  # 8 to 25
            for dealer_upcard in range(2, 12):  # 2 to 11 (Ace)
                # Create a synthetic state with simplified features
                # [player_sum, dealer_up_card, usable_ace, can_double]
                state = np.array([player_total, dealer_upcard, 0, 1])

                # Get basic strategy action
                if player_total <= 25 and player_total >= 8:
                    row_idx = min(player_total - 8, 17)  # Clip at 17+ (index 9)
                    col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                    basic_action = hard_totals[row_idx][col_idx]
                else:
                    basic_action = 0  # Default to hit for very high totals

                # Get agent action
                valid_actions = [0, 1]  # Hit and stand are always valid
                if state[3] == 1:  # Can double
                    valid_actions.append(2)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # Mask invalid actions
                masked_values = np.copy(action_probs)
                for i in range(len(action_probs)):
                    if i not in valid_actions:
                        masked_values[i] = -np.inf

                agent_action = np.argmax(masked_values)

                # Compare actions
                match = agent_action == basic_action

                # Update comparison matrix for visualization (1 for match, 0 for mismatch)
                if player_total <= 25 and player_total >= 8:
                    comparison_matrix[row_idx][col_idx] = 1 if match else 0

                # Print result for meaningful combinations (8-21)
                if 8 <= player_total <= 21:
                    print(f"{player_total:11d} | {dealer_upcard:13d} | {action_names[basic_action]:14s} | {action_names[agent_action]:14s} | {'✓' if match else '✗'}")

        # Plot hard totals comparison
        plt.subplot(3, 1, 1)
        plt.imshow(comparison_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 1], label='Match (1) vs Mismatch (0)')
        plt.title('Hard Totals: Agent vs Basic Strategy')
        plt.xlabel('Dealer Upcard (2-A)')
        plt.ylabel('Player Total (8-25+)')
        plt.xticks(range(10), ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'])
        plt.yticks(range(18), ['8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25+'])

        # 2. SOFT TOTALS
        print("\nSOFT TOTALS COMPARISON:")
        print("Player Hand | Dealer Upcard | Basic Strategy | Agent Decision | Match?")
        print("-" * 75)

        # Prepare matrix for visualization
        soft_comparison_matrix = np.zeros((9, 10))

        # Check each combination
        for ace_with in range(2, 11):  # A,2 through A,10
            for dealer_upcard in range(2, 12):  # 2 to 11 (Ace)
                player_total = 11 + ace_with  # A=11 + second card

                # Create a synthetic state with simplified features
                # [player_sum, dealer_up_card, usable_ace, can_double]
                state = np.array([player_total, dealer_upcard, 1, 1])

                # Get basic strategy action
                row_idx = ace_with - 2  # A,2 starts at index 0
                col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                basic_action = soft_totals[row_idx][col_idx]

                # Get agent action
                valid_actions = [0, 1]  # Hit and stand are always valid
                if state[3] == 1:  # Can double
                    valid_actions.append(2)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # Mask invalid actions
                masked_values = np.copy(action_probs)
                for i in range(len(action_probs)):
                    if i not in valid_actions:
                        masked_values[i] = -np.inf

                agent_action = np.argmax(masked_values)

                # Compare actions
                match = agent_action == basic_action

                # Update comparison matrix for visualization
                soft_comparison_matrix[row_idx][col_idx] = 1 if match else 0

                # Print result
                print(f"A,{ace_with:8d} | {dealer_upcard:13d} | {action_names[basic_action]:14s} | {action_names[agent_action]:14s} | {'✓' if match else '✗'}")

        # Plot soft totals comparison
        plt.subplot(3, 1, 2)
        plt.imshow(soft_comparison_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 1], label='Match (1) vs Mismatch (0)')
        plt.title('Soft Totals: Agent vs Basic Strategy')
        plt.xlabel('Dealer Upcard (2-A)')
        plt.ylabel('Player Hand')
        plt.xticks(range(10), ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'])
        plt.yticks(range(9), ['A,2', 'A,3', 'A,4', 'A,5', 'A,6', 'A,7', 'A,8', 'A,9', 'A,10'])

        # 3. PAIRS
        print("\nPAIRS COMPARISON (Note: Agent cannot split in current implementation):")
        print("Player Pair | Dealer Upcard | Basic Strategy | Agent Decision | Match?")
        print("-" * 75)

        # Prepare matrix for visualization
        pairs_comparison_matrix = np.zeros((10, 10))

        # For pairs, we'll just handle them as corresponding hard totals since our agent doesn't split
        for pair_card in range(2, 12):  # 2,2 through A,A
            for dealer_upcard in range(2, 12):  # 2 to 11 (Ace)
                if pair_card == 11:  # Ace pair
                    player_total = 12  # A=11 + A=1 (one ace is always counted as 1 to avoid bust)
                    usable_ace = 1
                else:
                    player_total = pair_card * 2
                    usable_ace = 0

                # Create a synthetic state with simplified features
                # [player_sum, dealer_up_card, usable_ace, can_double]
                state = np.array([player_total, dealer_upcard, usable_ace, 1])

                # Get basic strategy action
                row_idx = pair_card - 2  # 2,2 starts at index 0
                col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                basic_action = pairs[row_idx][col_idx]

                # Adjust for the fact our agent can't split
                adjusted_basic_action = basic_action
                if basic_action == 3:  # Split
                    # For pairs, default to hard total strategy if can't split
                    if pair_card == 11:  # A,A
                        adjusted_basic_action = 0  # Hit
                    elif pair_card == 5:  # 5,5
                        adjusted_basic_action = 2  # Double (treat as hard 10)
                    elif player_total >= 17:
                        adjusted_basic_action = 1  # Stand
                    else:
                        adjusted_basic_action = 0  # Hit

                # Get agent action
                valid_actions = [0, 1]  # Hit and stand are always valid
                if state[3] == 1:  # Can double
                    valid_actions.append(2)

                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # Mask invalid actions
                masked_values = np.copy(action_probs)
                for i in range(len(action_probs)):
                    if i not in valid_actions:
                        masked_values[i] = -np.inf

                agent_action = np.argmax(masked_values)

                # Compare actions (using adjusted basic action)
                match = agent_action == adjusted_basic_action

                # Update comparison matrix for visualization
                pairs_comparison_matrix[row_idx][col_idx] = 1 if match else 0

                # Print result
                pair_name = f"{pair_card},{pair_card}" if pair_card != 11 else "A,A"
                basic_action_name = "Split" if basic_action == 3 else action_names[basic_action]
                print(f"{pair_name:10s} | {dealer_upcard:13d} | {basic_action_name:14s} | {action_names[agent_action]:14s} | {'✓' if match else '✗'}")

        # Plot pairs comparison
        plt.subplot(3, 1, 3)
        plt.imshow(pairs_comparison_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(ticks=[0, 1], label='Match (1) vs Mismatch (0)')
        plt.title('Pairs: Agent vs Basic Strategy (Adjusted for No Split)')
        plt.xlabel('Dealer Upcard (2-A)')
        plt.ylabel('Player Pair')
        plt.xticks(range(10), ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A'])
        plt.yticks(range(10), ['2,2', '3,3', '4,4', '5,5', '6,6', '7,7', '8,8', '9,9', '10,10', 'A,A'])

        plt.tight_layout()
        plt.savefig('blackjack_strategy_comparison.png')
        plt.show()

        # Calculate and print overall accuracy
        hard_accuracy = np.mean(comparison_matrix)
        soft_accuracy = np.mean(soft_comparison_matrix)
        pairs_accuracy = np.mean(pairs_comparison_matrix)

        print(f"\nACCURACY SUMMARY:")
        print(f"Hard Totals: {hard_accuracy:.2%}")
        print(f"Soft Totals: {soft_accuracy:.2%}")
        print(f"Pairs: {pairs_accuracy:.2%}")
        print(f"Overall: {np.mean([hard_accuracy, soft_accuracy, pairs_accuracy]):.2%}")

        return hard_accuracy, soft_accuracy, pairs_accuracy
    
    compare_with_basic_strategy(agent)

