import ctypes
import functools
import operator
from ctree.c.nodes import SymbolRef, Mul, Add, Constant, Cast
from ctree.cpp.nodes import CppDefine

import ast
import functools

import numpy as np
import sympy
from snowflake.compiler_nodes import IndexOp, Space, NDSpace
from snowflake.vector import Vector

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
    frame = IndexOp(elts)

    for component in index:
        if isinstance(component, int):
            elts.append(ast.Num(n=component))
        elif isinstance(component, sympy.Expr):
            elts.append(sympy_to_ast(component))
    return frame


def fill_iteration_spaces(node, shape):
    """Changes Spaces from relative to absolute bounds"""
    class IterationSpaceFiller(ast.NodeTransformer):
        def visit_Space(self, node):
            node = self.generic_visit(node)
            node.low = Vector(
                [i + size if i < 0 else i for i, size in zip(node.low, shape)]
            )

            node.high = Vector(
                [i + size if i <= 0 else i for i, size in zip(node.high, shape)]
            )
            return node

            # new_spaces = []
            # for space, size in zip(node.spaces, shape):
            #     low, high, stride = space
            #     if low < 0:  # low = 0 means start at bottom, low = -1 means start at top
            #         low = size + low
            #
            #     if high < 0:  # high = 0 means end at top
            #         high = size + high
            #
            #     new_spaces.append(Space(low, high, stride))
            # return NDSpace(new_spaces)
    return IterationSpaceFiller().visit(node)


def calculate_ND_volume(ndspace):
    if not all(all(i >= 0 for i in space.low) and all(i > 0 for i in space.high) for space in ndspace.spaces):
        raise ValueError("All dimensions must have their actual sizes filled in.")
    strided_volume = 1
    total_volume = 1
    for space in ndspace.spaces:
        total_volume *= space.high - space.low
        strided_volume *= (space.high - space.low) // space.stride
    return functools.reduce(operator.mul, total_volume), functools.reduce(operator.mul, strided_volume)