import time


def timed(func):
    """ Decorator method to return timing of function """
    def wrap(*args, **kwargs):
        start = time.time()
        wrapped_func = func(*args, **kwargs)
        elapsed = time.time() - start
        print '%s.%s took %0.3f ms' % (func.__module__, func.func_name, elapsed * 1000)
        return wrapped_func
    return wrap
