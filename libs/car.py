import cv2 as cv
import threading
import time
import requests
import numpy as np
from queue import Queue


class Car:
    def __init__(self, host):
        self.host = host
        self.stream_port = 8080
        self.cmd_port = 8000

    def get_stream_url(self):
        return self.host + ":" + str(self.stream_port) + '/?action=stream'

    def get_cmd_url(self, action):
        return self.host + ":" + str(self.cmd_port) + '/run/?action=' + action

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
        url = self.get_stream_url()
        q = Queue(2)  # Buffer only 2 frames
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        play_thread = threading.Thread(target=self.play, args=(q, 1/rate,))
        buffer_thread.start()
        play_thread.start()

    def get_snapshot(self):
        url = self.get_stream_url()
        q = Queue(2)
        stream = cv.VideoCapture(url)
        buffer_thread = threading.Thread(target=self.buffer, args=(q, stream,))
        buffer_thread.start()
        return q

    def start(self):
        self.run_action("forward")

    def stop(self):
        self.run_action("stop")

    def left(self):
        self.run_action("fwleft")

    def right(self):
        self.run_action("fwright")

    def straight(self):
        self.run_action("fwstraight")

    def run_action(self, action):
        # bwready | forward | backward | stop
        # fwready | fwleft | fwright |  fwstraight
        url = self.get_cmd_url(action)
        requests.get(url)
