from better_environment import BlackjackEnv


env = BlackjackEnv(count_type="empty")

state, reward, done = env.reset()
print(state, reward, done)
while True:
    action = int(input())
    if action == 4:
        state, reward, done = env.reset()
        print("==================RESET===================")
        print(state, reward, done)
        continue
    if action == 5:
        rewards = env.deliver_rewards()
        print(rewards)
        continue
    
    state, reward, done = env.step(action)
    print(state, reward, done)