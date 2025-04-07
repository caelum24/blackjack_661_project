from grid_search import perform_grid_search

deck_counts      = [1, 2]     
gammas           = [0.90, 0.95]  
batch_sizes      = [32, 64] 
learning_rates   = [0.001, 0.0005]
epsilons_starts   = [1.0, 0.5]
epsilon_decays   = [0.999, 0.995]
epsilon_mins     = [0.05] 
min_bets         = [1, 5, 10]
max_bets         = [50, 100, 200]
initial_bankrolls = [500, 1000, 5000]
memory_sizes     = [5000, 10000, 20000]

perform_grid_search(deck_counts,gammas,batch_sizes,learning_rates,epsilons_starts,
                    epsilon_decays,epsilon_mins,min_bets,max_bets,initial_bankrolls,memory_sizes)