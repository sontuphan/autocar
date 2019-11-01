import cv2 as cv
import threading
import time
import numpy as np
from queue import Queue


class Car:
    def __init__(self, host, port=8080):
        self.host = host
        self.port = port

    def get_url(self, action):
        return self.host + ":" + str(self.port) + '/?action=' + action

    def play(self, q, sec):
        while True:
            time.sleep(sec)
            frame = q.get()
            cv.imshow("car-camera-tups", frame)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        cv.destroyWindow("car-camera-tups")

    def buffer(self, q, stream):
        while stream.isOpened():
            ret, frame = stream.read()
            if ret is not True:
                break
            if q.full():
                q.get()
            q.put(frame)

    def get_camera(self, rate):
        url = self.get_url('stream')
        q = Queue(2)  # Buffer only 2 frames
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        play_thread = threading.Thread(target=self.play, args=(q, 1/rate,))
        buffer_thread.start()
        play_thread.start()

    def get_snapshot(self):
        url = self.get_url('stream')
        q = Queue(5)
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        buffer_thread.start()
        return q
