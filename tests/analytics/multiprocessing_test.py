from __future__ import print_function
from multiprocessing import Pool, cpu_count
import itertools

__author__ = 'nzhang-dev'


class obj(object):
    def __init__(self, x):
        self.x = x

def f((a, b)):
    return a.x * b.x

if __name__ == "__main__":
    objects = [obj(i) for i in range(32)]
    p = Pool(cpu_count())
    res = p.map(f, itertools.product(objects, repeat=2))
    print(res)