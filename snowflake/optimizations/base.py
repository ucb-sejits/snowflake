from __future__ import print_function, division
import abc
import ast

__author__ = 'nzhang-dev'


class Optimization(object):
    optimization_level = -1 # undefined

    @abc.abstractproperty
    def tuning_driver(self):
        pass

    @abc.abstractmethod
    def optimize(self, tree):
        pass


class StencilOptimization(Optimization):
    """Optimizations that happen at the Stencil level"""

class CompilerOptimization(Optimization):
    """Optimizations that happen at the CompilerNode level"""
