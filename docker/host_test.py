import time

import ray


# Wait for clients to connect and print available resources
ray.init()
start = time.perf_counter()
curtime = start
while curtime - start < 100:
    print(ray.available_resources())
    time.sleep(10)
    curtime = time.perf_counter()
