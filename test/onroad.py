import time
from libs import rl, car


def test():
    # Setup car
    HOST = "http://172.31.0.29"
    picar = car.Car(HOST)
    # Setup Markov Decision Process
    mdp = rl.MDP(picar, 1)

    picar.start()
    picar.speed(1)

    prev_state = None
    prev_action = None
    prev_value = None
    while True:
        frame = mdp.extract_frame()
        current_state = mdp.get_state(frame)
        if prev_state is not None:
            mdp.learn(prev_state, prev_action, prev_value, current_state)
        next_action, value = mdp.get_action(current_state)
        if next_action == -1:
            print("left")
            picar.left()
        elif next_action == 0:
            print("straight")
            picar.straight()
        else:
            print("right")
            picar.right()
        # send cmd to car here
        prev_state = current_state
        prev_action = next_action
        prev_value = value
        # Learning step
        time.sleep(0.2)
