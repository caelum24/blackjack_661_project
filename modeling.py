import numpy as np
# import random
import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import deque
import matplotlib.pyplot as plt
# from environment import BlackjackEnv 

def model(agent):
    '''
    Wrapper to create the strategy table and compare agent results
    Pass the trained agent and this module will print the strategy comparisons
    '''
        
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_basic_strategy_table():
        """Create blackjack basic strategy tables"""
        # Hard totals basic strategy (rows=player total, cols=dealer upcard)
        # 0=Hit, 1=Stand, 2=Double, 3=Split
        hard_totals = np.zeros((14, 10), dtype=int) # NOTE -> 14 states from 8 to 21

        # Fill in the basic strategy for hard totals (8-17 vs 2-Ace)
        ###### NOTE -> this is for DEALER STANDS ON SOFT 17 GAMES ######

        # Row 0 = hard 8, Row 1 = hard 9, ..., Row 9 = hard 17
        # Col 0 = dealer 2, Col 1 = dealer 3, ..., Col 9 = dealer Ace

        ######## HARD TOTALS ########
        # Always hit for hard 8 or less
        hard_totals[0, :] = 0  # Hard 8 - always hit

        # Hard 9
        hard_totals[1, 1:5] = 2  # Double vs 3-6

        # Hard 10
        hard_totals[2, 0:8] = 2  # Double vs 2-9
        # otherwise hit

        # Hard 11
        hard_totals[3, :9] = 2  # Double vs everything but A

        # Hard 12
        hard_totals[4, 2:5] = 1  # Stand vs 4-6

        # Hard 13-16
        for i in range(5, 9):
            hard_totals[i, 0:5] = 1  # Stand vs 2-6

        # Hard 17+
        for i in range(9, 14):
            hard_totals[i, :] = 1  # Always stand

        ###### SOFT TOTALS ########

        # Soft totals basic strategy (A,2 through A,10)
        # 0=Hit, 1=Stand, 2=Double, 3=Split

        soft_totals = np.zeros((8, 10), dtype=int)

        # A,2 and A,3 (13-14)
        soft_totals[0:2, 3:5] = 2  # Double vs 5-6

        # A,4 and A,5 (15-16)
        soft_totals[2:4, 2:5] = 2  # Double vs 4-6

        # A,6 (17)
        soft_totals[4, 1:5] = 2  # Double vs 3-6

        # A,7 (18)
        soft_totals[5, 0] = 1  # Stand vs 2
        soft_totals[5, 5:7] = 1  # Stand vs 7-8
        soft_totals[5, 1:5] = 2  # Double vs 3-6
        
        # A,8 and A,9 (19-20)
        soft_totals[6:, :] = 1  # Always stand

        ###### Pairs basic strategy #######

        # 0=Hit, 1=Stand, 2=Double, 3=Split
        # NOTE -> double after split IS allowed in our environment
        pairs = np.zeros((10, 10), dtype=int)

        # 2,2 and 3,3
        pairs[0:2, 0:6] = 3  # Split vs 2-7
        # hit otherwise
        
        # 4,4
        pairs[2, 3:5] = 3  # Split vs 5-6
        # hit otherwise

        # 5,5 - treat as hard 10
        pairs[3, 0:8] = 2  # Double vs 2-9
        # otherwise hit

        # 6,6
        pairs[4, 0:5] = 3  # Split vs 2-6
        # otherwise hit

        # 7,7
        pairs[5, 0:6] = 3  # Split vs 2-7
        # otherwise hit

        # 8,8
        pairs[6, :] = 3  # Always split

        # 9,9
        pairs[7, 0:5] = 3  # Split vs 2-6
        pairs[7, 5] = 1  # Stand vs 7
        pairs[7, 6:8] = 3   # Split vs 8-9
        pairs[7, 8:] = 1  # Stand vs 10-A

        # 10,10
        pairs[8, :] = 1  # Always stand

        # A,A
        pairs[9, :] = 3  # Always split

        return hard_totals, soft_totals, pairs

    def compare_with_basic_strategy(agent):
        """Compare agent decisions with basic strategy for all combinations"""
        hard_totals, soft_totals, pairs = create_basic_strategy_table()

        # Create mapping for action names
        action_names = {0: "Hit", 1: "Stand", 2: "Double", 3: "Split"}

        # Prepare the results
        print("\n\n===== COMPARING AGENT VS BASIC STRATEGY =====\n")

        # Create a figure for the comparison visualization
        plt.figure(figsize=(18, 15))

        # 1. HARD TOTALS
        print("\nHARD TOTALS COMPARISON:")
        print("Player Total | Dealer Upcard | Basic Strategy | Agent Decision | Match?")
        print("-" * 75)

        # Prepare matrix for visualization
        comparison_matrix = np.zeros((14, 10)) # 14 for 8-21

        # Check each combination
        for player_total in range(8, 22):  # 8 to 21
            for dealer_upcard in range(2, 12):  # 2 to 11 (Ace)
                # Create a synthetic state with simplified features
                # [player_sum, dealer_up_card, usable_ace, can_double, can_split]
                #TODO -> full will error out
                state = np.array([player_total, dealer_upcard, 0, 1, 0])

                # Add card counting information based on agent's count_type
                if agent.count_type == "full":
                    # For full count, add card percentages for each card value
                    card_percentages = np.ones(10) * 1/13  # Equal distribution for comparison (10 gets 4x because 10,J,Q,K)
                    card_percentages[-2] = 4/13 # 10 gets 4x the probability in general
                elif agent.count_type != "empty":
                    # For system counts (hi_lo, zen, uston_apc, ten_count), add normalized count
                    state = np.concatenate([state, [0.0]])  # Neutral count for comparison

                # Get basic strategy action
                if player_total <= 21 and player_total >= 8:
                    row_idx = min(player_total - 8, 17)  # Clip at 17+ (index 9)
                    col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                    basic_action = hard_totals[row_idx][col_idx]
                else:
                    basic_action = 0  # Default to hit for very high totals

                # get the agent to act
                agent_action = agent.act(state, eval=True)
                # # Get agent action
                # valid_actions = [0, 1]  # Hit and stand are always valid
                # if state[3] == 1:  # Can double
                #     valid_actions.append(2)
                # if len(state) > 4 and state[4] == 1:
                #     valid_actions.append(3)

                # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                # with torch.no_grad():
                #     action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # # Mask invalid actions
                # masked_values = np.copy(action_probs)
                # for i in range(len(action_probs)):
                #     if i not in valid_actions:
                #         masked_values[i] = -np.inf

                # agent_action = np.argmax(masked_values)

                # Compare actions
                match = (agent_action == basic_action)

                # Update comparison matrix for visualization (1 for match, 0 for mismatch)
                if player_total <= 21 and player_total >= 8:
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
        plt.yticks(range(14), ['8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'])

        # 2. SOFT TOTALS
        print("\nSOFT TOTALS COMPARISON:")
        print("Player Hand | Dealer Upcard | Basic Strategy | Agent Decision | Match?")
        print("-" * 75)

        # Prepare matrix for visualization
        soft_comparison_matrix = np.zeros((8, 10))

        # Check each combination
        for ace_with in range(2, 10):  # A,2 through A,9
            for dealer_upcard in range(2, 12):  # 2 to 11 (Ace)
                player_total = 11 + ace_with  # A=11 + second card

                # Create a synthetic state with simplified features
                # [player_sum, dealer_up_card, usable_ace, can_double, can_split]
                state = np.array([player_total, dealer_upcard, 0, 1, 0])

                # Add card counting information based on agent's count_type
                if agent.count_type == "full":
                    # For full count, add card percentages for each card value
                    card_percentages = np.ones(10) * 0.1  # Equal distribution for comparison
                    state = np.concatenate([state, card_percentages])
                elif agent.count_type != "empty":
                    # For system counts (hi_lo, zen, uston_apc, ten_count), add normalized count
                    state = np.concatenate([state, [0.0]])  # Neutral count for comparison

                # Get basic strategy action
                row_idx = ace_with - 2  # A,2 starts at index 0
                col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                basic_action = soft_totals[row_idx][col_idx]


                agent_action = agent.act(state, eval=True)

                # # Get agent action
                # valid_actions = [0, 1]  # Hit and stand are always valid
                # if state[3] == 1:  # Can double
                #     valid_actions.append(2)
                # if len(state) > 4 and state[4] == 1:
                #     valid_actions.append(3)

                # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                # with torch.no_grad():
                #     action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # # Mask invalid actions
                # masked_values = np.copy(action_probs)
                # for i in range(len(action_probs)):
                #     if i not in valid_actions:
                #         masked_values[i] = -np.inf

                # agent_action = np.argmax(masked_values)

                # Compare actions
                match = (agent_action == basic_action)

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
                # [player_sum, dealer_up_card, usable_ace, can_double, can_split]
                state = np.array([player_total, dealer_upcard, usable_ace, 1, 1])

                # Add card counting information based on agent's count_type
                if agent.count_type == "full":
                    # For full count, add card percentages for each card value
                    card_percentages = np.ones(10) * 1/13  # Equal distribution for comparison (10 gets 4x because 10,J,Q,K)
                    card_percentages[-2] = 4/13
                    state = np.concatenate([state, card_percentages])
                elif agent.count_type != "empty":
                    # For system counts (hi_lo, zen, uston_apc, ten_count), add normalized count
                    state = np.concatenate([state, [0.0]])  # Neutral count for comparison

                # Get basic strategy action
                row_idx = pair_card - 2  # 2,2 starts at index 0
                col_idx = min(dealer_upcard - 2, 9)  # Convert dealer card to index
                basic_action = pairs[row_idx][col_idx]

                # NOTE -> adjusted basic action can go away because we have successfully implemented splitting
                # # Adjust for the fact our agent can't split
                # adjusted_basic_action = basic_action
                # if basic_action == 3:  # Split
                #     # For pairs, default to hard total strategy if can't split
                #     if pair_card == 11:  # A,A
                #         adjusted_basic_action = 0  # Hit
                #     elif pair_card == 5:  # 5,5
                #         adjusted_basic_action = 2  # Double (treat as hard 10)
                #     elif player_total >= 17:
                #         adjusted_basic_action = 1  # Stand
                #     else:
                #         adjusted_basic_action = 0  # Hit

                agent_action = agent.act(state, eval=True)
                # # Get agent action
                # valid_actions = [0, 1]  # Hit and stand are always valid
                # if state[3] == 1:  # Can double
                #     valid_actions.append(2)
                # if len(state) > 4 and state[4] == 1:
                #     valid_actions.append(3)

                # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                # with torch.no_grad():
                #     action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]

                # # Mask invalid actions
                # masked_values = np.copy(action_probs)
                # for i in range(len(action_probs)):
                #     if i not in valid_actions:
                #         masked_values[i] = -np.inf

                # agent_action = np.argmax(masked_values)

                # Compare actions (using adjusted basic action)
                match = (agent_action == basic_action)

                # Update comparison matrix for visualization
                pairs_comparison_matrix[row_idx][col_idx] = 1 if match else 0

                # Print result
                pair_name = f"{pair_card},{pair_card}" if pair_card != 11 else "A,A"
                basic_action_name = action_names[basic_action]
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
    
    def visualize_action_preferences(agent):
        """Visualize the model's action preferences for specific game states"""
        # Define the test cases
        test_states = [
            # Player 8,8 vs Dealer 10
            np.array([16, 10, 0, 1, 1]),  # [player_sum, dealer_up_card, usable_ace, can_double, can_split]
            # Player 10,10 vs Dealer 2
            np.array([20, 2, 0, 1, 1]),
            # Player 9,2 vs Dealer 3
            np.array([11, 3, 0, 1, 0])    # 9 + 2 = 11
        ]
        
        # Add card counting information based on agent's count_type
        for i in range(len(test_states)):
            if agent.count_type == "full":
                # For full count, add card percentages for each card value
                card_percentages = np.ones(10) * 1/13  # Equal distribution for comparison (10 gets 4x because 10,J,Q,K)
                card_percentages[-2] = 4/13 # 10 gets 4x the probability in general
            elif agent.count_type != "empty":
                # For system counts (hi_lo, zen, uston_apc, ten_count), add normalized count
                test_states[i] = np.concatenate([test_states[i], [0.0]])  # Neutral count for comparison
        
        state_names = [
            "Player 8,8 vs Dealer 10",
            "Player 10,10 vs Dealer 2",
            "Player 9,2 vs Dealer 3"
        ]
        
        action_names = ["Hit", "Stand", "Double", "Split"]
        
        # Create a figure for the visualization
        plt.figure(figsize=(15, 5))
        
        # Process each test state
        for i, (state, name) in enumerate(zip(test_states, state_names)):

            # Determine valid actions to ensure model doesn't do something illegal
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)
            if len(state) > 4 and state[4] == 1:
                valid_actions.append(3)

            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            
            # Get action probabilities
            with torch.no_grad():
                action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]
            
            # NOTE -> added this to filter model to give no value to invalid actions
            # Filter out invalid actions by setting their values to a very low number
            masked_values = np.copy(action_probs)
            for k in range(len(action_names)):
                if k not in valid_actions:
                    masked_values[k] = -np.inf

            action_probs = masked_values

            # Apply softmax to get probabilities
            action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
            
            # Create subplot
            plt.subplot(1, 3, i+1)
            bars = plt.bar(action_names, action_probs)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.title(name)
            plt.ylabel('Probability')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('action_preferences.png')
        plt.show()

    
    def visualize_count_effect(agent, player_hand, dealer_hand, usable_ace, can_double, can_split):
        """Visualize the model's action preferences for specific game states, sweeping over different counts
            This is meant for counts where there is only 1 count involved
        """
        count_type = agent.count_type
        action_names = ["Hit", "Stand", "Double", "Split"]
        
        # Create a figure for the visualization
        plt.figure(figsize=(10, 10))

        
        
        # Process each test state
        for i, system_count in enumerate(range(-10, 10)): # sweep over 20 different counts

            state = np.array([player_hand, dealer_hand, usable_ace, can_double, can_split])

            # Determine valid actions to ensure model doesn't do something illegal
            valid_actions = [0, 1]  # Hit and stand are always valid
            if state[3] == 1:  # Can double
                valid_actions.append(2)
            if len(state) > 4 and state[4] == 1:
                valid_actions.append(3)

            state = np.concatenate([state, np.array([system_count])])

            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            # print(state_tensor)
            # Get action probabilities
            with torch.no_grad():
                action_probs = agent.policy_net(state_tensor, training=False).cpu().data.numpy()[0]
            
            # NOTE -> added this to filter model to give no value to invalid actions
            # Filter out invalid actions by setting their values to a very low number
            masked_values = np.copy(action_probs)
            for k in range(len(action_names)):
                if k not in valid_actions:
                    masked_values[k] = -np.inf

            action_probs = masked_values

            # Apply softmax to get probabilities
            action_probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
            
            # Create subplot -> 20 values in 4 rows of 5
            plt.subplot(4, 5, i+1)
            bars = plt.bar(action_names, action_probs)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom')
            
            plt.title(f"{str(state[:-1])}, {count_type} = {system_count}")
            plt.ylabel('Probability')
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('action_preferences_over_count_range.png')
        plt.show()
    
    compare_with_basic_strategy(agent)
    visualize_action_preferences(agent)
    # visualize_count_effect(agent, 17, 10, 0, 1, 0)


