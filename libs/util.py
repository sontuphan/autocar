import math
from datetime import datetime
from random import seed, randint
import numpy as np


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    x_1, y_1, x_2, y_2 = v1
    a_1, b_1, a_2, b_2 = v2
    v_1 = [x_1-x_2, y_1-y_2]
    v_2 = [a_1-a_2, b_1-b_2]
    radian = math.acos(dotproduct(v_1, v_2) / (length(v_1) * length(v_2)))
    degree = np.degrees(radian)
    return degree


def random(maximum):
    seed(datetime.now())
    return randint(0, maximum)
