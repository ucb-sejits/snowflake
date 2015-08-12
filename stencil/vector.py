from itertools import izip_longest
from numbers import Number
import operator
import itertools
from utils import partition, shell

__author__ = 'nzhang-dev'


class Vector(tuple):
    """
    A Vector that supports arithmetic. These can be used for numpy indicies and has neighborhood utility functions.
    """
    @classmethod
    def unit_vector(cls, dim, ndim):
        vec = [0 for i in range(ndim)]
        vec[dim] = 1
        return cls(vec)

    @classmethod
    def zero_vector(cls, ndim):
        return cls((0,)*ndim)

    @classmethod
    def von_neumann_vectors(cls, ndim, radius=1, closed=False):
        """
        :param ndim: Number of dimensions
        :param radius: Von Neumann Neighborhood radius
        :param closed: Should the neighborhood include all points r < radius or only r = radius
        :return: Iterator of vectors in the Von Neumann neighborhood, :math:`\{v | v \in Z^n, ||v||_\infty = r\}`
        """
        neighborhood = (cls(i) for i in partition(radius, ndim))
        if closed and radius > 1:
            return itertools.chain(neighborhood, cls.von_neumann_vectors(ndim, radius-1, closed))
        return neighborhood

    @classmethod
    def moore_vectors(cls, ndim, radius=1, closed=False):
        """
        :param ndim: Number of dimensions
        :param radius: Moore Neighborhood radius
        :param closed: Should the neighborhood include all points r < radius or only r = radius
        :return: Iterator of vectors in the Moore neighborhood, :math:`\{v | v \in Z^n, ||v||_1 = r\}`
        """
        neighborhood = (cls(i) for i in shell(radius, ndim))
        if closed and radius > 1:
            return itertools.chain(neighborhood, cls.moore_vectors(ndim, radius-1, closed))
        return neighborhood

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
