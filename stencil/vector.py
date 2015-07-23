from itertools import izip_longest
from numbers import Number
import operator

__author__ = 'nzhang-dev'


class Vector(tuple):
    @classmethod
    def __apply(cls, a, b, op):
        if isinstance(b, Number):
            b = (b,) * len(a)
        elif isinstance(a, Number):
            a = (a,) * len(b)
        return cls(op(i, j) for i, j in izip_longest(a, b, fillvalue=0))

    def __add__(self, other):
        return Vector.__apply(self, other, operator.add)

    def __sub__(self, other):
        return Vector.__apply(self, other, operator.sub)

    def __mul__(self, other):
        return Vector.__apply(self, other, operator.mul)

    def __div__(self, other):
        return Vector.__apply(self, other, operator.div)