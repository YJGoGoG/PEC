import os
# An example
for i in range(10):
    str1 = ('python main.py --policy PEC --if_random True --num_q 20 --critic_number 2 --env Walker2d-v3 --seed ' + str(i))
    p = os.system(str1)
    print(p)



