import cv2 as cv
from libs import rl, car


def test_mdp():
    cap = cv.VideoCapture("data/realrun3.mp4")
    mdp = rl.MDP(0.1, cap)

    prev_state = None
    prev_action = None
    prev_value = None
    while cap.isOpened():
        current_state = mdp.get_state()
        if prev_state is not None:
            mdp.learn(prev_state, prev_action, prev_value, current_state)
        next_action, value = mdp.get_action(current_state)
        if next_action == -1:
            print("left")
        elif next_action == 0:
            print("straight")
        else:
            print("right")
        # send cmd to car here
        prev_state = current_state
        prev_action = next_action
        prev_value = value
