# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 23:09:46 2021

@author: zeids
"""
import multiprocessing
import time
def dummy(itr):
    for i in range(itr):
        print(i)
        time.sleep(1)
if __name__ == '__main__':
    proc = multiprocessing.Process(target=dummy, args=(100,))
    proc.start()
    time.sleep(5)
    proc.terminate()  # sends a SIGTERM
    #proc.join()
