import time

def a():
    time.sleep(1) # some computation...

def b():
    a()
    time.sleep(2) # more computation...

b()
