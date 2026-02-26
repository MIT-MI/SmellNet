import os
import time


def toggle():
    state = 0
    state_mapping = {0: 'ambient', 1: 'substance'}

    file = 'state.txt'
    # write the initial state
    with open(file, 'w') as f:
        f.write(str(state))
    current_time = time.time()
    print(f"current state: {state_mapping[state]}")
    while True:
        print(f'Current state: {state_mapping[state]}')
        user_input = input('Toggle state?: ')
        state = 1 - state
        with open(file, 'w') as f:
            f.write(str(state))
        # if time.time() - current_time > 20:
        #     state = 1 - state
        #     current_time = time.time()
        #     with open(file, 'w') as f:
        #         f.write(str(state))
        #     print(f"current state: {state_mapping[state]}")


if __name__ == '__main__':
    toggle()