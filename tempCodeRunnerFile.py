        while done != 2:

            action = agent.act(state)
            next_state, _, done = env.step(action)

            state = next_state