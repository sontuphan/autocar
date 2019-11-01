import cv2 as cv
import matplotlib.pyplot as plt
from libs import car, utils
import time

HOST = "http://172.31.0.29"
picar = car.Car(HOST)


def test_camera():
    picar.get_camera(24)


def test_snapshot():
    buffer = picar.get_snapshot()
    while True:
        plt.imshow(buffer.get())
        plt.show(block=False)
        plt.pause(0.01)
