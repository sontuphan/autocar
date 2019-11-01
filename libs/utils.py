import threading


def set_interval(func, args, sec):
    def func_wrapper():
        set_interval(func, args, sec)
        func(args)
    t = threading.Timer(sec, func_wrapper)
    t.start()
    return t
