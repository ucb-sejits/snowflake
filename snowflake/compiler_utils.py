import ctypes
import functools
from ctree.c.nodes import SymbolRef, Mul, Add, Constant, Cast
from ctree.cpp.nodes import CppDefine

import ast
import functools

import numpy as np
import sympy

__author__ = 'nzhang-dev'



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

def sympy_to_ast(exp):
    if isinstance(exp, sympy.Symbol):
        return ast.Name(id=exp.name, ctx=ast.Load())

    if isinstance(exp, (sympy.Integer, int)):
        return ast.Num(n=int(exp))

    if isinstance(exp, sympy.Rational):
        return ast.BinOp(sympy_to_ast(exp.p), ast.Div(), sympy_to_ast(exp.q))

    func_map = {
        sympy.Add: ast.Add,
        sympy.Mul: ast.Mult,
    }
    if type(exp) in func_map:
        res = reduce(lambda x, y: ast.BinOp(x, func_map[type(exp)](), y), map(sympy_to_ast, exp.args))
        return res

    raise Exception("Can't handle {} of type {}".format(exp, type(exp)))

def index_to_ast(index):
    elts = []
    frame = ast.Tuple(elts=elts, ctx=ast.Load())

    for component in index:
        if isinstance(component, int):
            elts.append(ast.Num(n=component))
        elif isinstance(component, sympy.Expr):
            elts.append(sympy_to_ast(component))
    return frame