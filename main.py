import random
from environment import Environment

random.seed(1)

if __name__ == '__main__':

    env = Environment() 

    actions = ['up', 'down', 'left', 'right']

    max_iter = 100

    i = 0
    while i < max_iter:
        action = random.choices(actions, weights=[1, 1, 1, 1], k=1)[0]
        reward, state, done = env.step(action)

        print(state)

        if done:
            break

        i +=1

