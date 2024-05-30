import cython_verlet
import time

start_time = time.time()
cython_verlet.main(nn=9)
end_time = time.time()

print("Time：", end_time - start_time)
