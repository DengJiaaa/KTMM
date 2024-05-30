import Ncython
import time

start_time = time.time()
Ncython.main(nn=200)
end_time = time.time()

print("Time：", end_time - start_time)