import ctypes
import functools
from ctree.c.nodes import SymbolRef, Mul, Add, Constant, Cast
from ctree.cpp.nodes import CppDefine

__author__ = 'nzhang-dev'

import numpy as np

def get_output_name():
    pass

def generate_encode_macro(name, shape):
    multipliers = [1] + list(np.cumproduct(shape[::-1]))[:-1]
    multipliers = [Constant(i) for i in reversed(multipliers)]
    params = [SymbolRef(name='x{}'.format(dim)) for dim in range(len(shape))]
    variables = []
    for var in params:
        cp = var.copy()
        cp._force_parentheses = True
        variables.append(Cast(ctypes.c_long(), cp))

    products = [Mul(mult, var) for mult, var in zip(multipliers, variables)]
    total = functools.reduce(Add, products)
    return CppDefine(name=name, params=params, body=total)