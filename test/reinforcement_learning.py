import cv2 as cv
import numpy as np
from libs import rl


def test_reward_function():
    cap = cv.VideoCapture("data/realrun3.mp4")
    if cap.isOpened():
        frame = cap.read()[1]
        reward = rl.reward(frame)
        print(reward)

    cap.release()
    cv.destroyAllWindows()
