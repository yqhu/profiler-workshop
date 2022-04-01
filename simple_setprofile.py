import time
import sys

def a():
    time.sleep(1) # some computation...

def b():
    a()
    time.sleep(2) # more computation...

frames = []
def profiler(frame, event, arg):
    t = time.time()
    if event == 'call' or event == 'return':
        frames.append((frame, event, t))

sys.setprofile(profiler)
b()
sys.setprofile(None)

for frame, event, t in frames:
    print(frame.f_code.co_name, event, t)
