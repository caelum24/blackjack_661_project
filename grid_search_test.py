from grid_search import perform_grid_search

deck_counts      = [6]     
gammas           = [0.90, 0.95, 0.99]  
batch_sizes      = [32, 64, 128] 
learning_rates   = [0.0001, 0.0005, 0.001]
epsilons_starts   = [1.0, 0.5]
epsilon_decays   = [0.98, 0.99, 0.995]
epsilon_mins     = [0.01, 0.001] 
min_bets         = [1]
max_bets         = [50]
initial_bankrolls = [1000]
memory_sizes     = [1000, 20000, 50000]

perform_grid_search(deck_counts,gammas,batch_sizes,learning_rates,epsilons_starts,
                    epsilon_decays,epsilon_mins,min_bets,max_bets,initial_bankrolls,memory_sizes)