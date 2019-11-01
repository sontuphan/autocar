from queue import Queue
from threading import Thread
import time

# A thread that produces data


def producer(out_q):
    counter = 0
    while True:
        if out_q.full():
            out_q.get()
        out_q.put(counter)
        counter += 1

# A thread that consumes data


def consumer(in_q):
    while True:
        counter = in_q.get()
        print(counter)
        time.sleep(1)


# Create the shared queue and launch both threads
q = Queue(5)
t1 = Thread(target=consumer, args=(q, ))
t2 = Thread(target=producer, args=(q, ))
t1.start()
t2.start()
