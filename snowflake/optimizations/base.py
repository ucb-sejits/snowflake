from __future__ import print_function, division
import abc
import ast

__author__ = 'nzhang-dev'


class Optimization(object):
    optimization_level = -1 # undefined

    @abc.abstractmethod
    def get_tuning_driver(self, shape):
        pass

    @abc.abstractmethod
    def optimize(self, tree, params):
        pass
