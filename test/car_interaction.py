import cv2 as cv
import matplotlib.pyplot as plt
from libs import car
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


def test_action():
    picar.start()
    time.sleep(1)
    picar.left()
    time.sleep(1)
    picar.straight()
    time.sleep(1)
    picar.right()
    time.sleep(1)
    picar.stop()


def test_speed():
    picar.start()
    time.sleep(3)
    picar.speed(4)
    time.sleep(3)
    picar.stop()


def test_general():
    test_camera()
    test_action()
