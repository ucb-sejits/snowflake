import copy
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
from snowflake.nodes import SparseWeightArray, StencilOp
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
    return ast.parse(str(exp)).body[0].value

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

            # now we clip the high to fit perfectly. For example, 1 to 10 by 3 should really be just 1, 4, 7 (so 1 to 8)
            # so really the upper bound should be upper - ((upper - lower - 1) % stride)
            node.high = Vector(
                [h - ((h - l) % s) for h, l, s in zip(node.high, node.low, node.stride)]
            )
            return node

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

def is_homogenous_space(ndspace):
    if not ndspace.spaces:
        raise ValueError("Empty spaces are not allowed")
    if any(space.stride != ndspace.spaces[0].stride for space in ndspace.spaces):
        return False

    #offsets in each dimension must be less than stride apart for start and end
    low_low = ndspace.spaces[0].low
    low_high = ndspace.spaces[0].low
    high_low = ndspace.spaces[0].high
    high_high = ndspace.spaces[0].high
    strides = ndspace.spaces[0].stride
    for space in ndspace.spaces:
        low_low = tuple(min(i, j) for i, j in zip(low_low, space.low))
        low_high = tuple(max(i, j) for i, j in zip(low_high, space.low))
        high_low = tuple(min(i, j) for i, j in zip(high_low, space.high))
        high_high = tuple(max(i, j) for i, j in zip(high_high, space.high))
    return all(
        all((high - low) <= stride for stride, low, high in zip(strides, low, high))
        for low, high in ((low_low, low_high), (high_low, high_high))
    )


class StencilShifter(ast.NodeTransformer):
    def __init__(self, offset):
        self.offset = offset

    def visit_FunctionCall(self, node):
        if not node.func.name.startswith("encode"):
            return self.generic_visit(node)
        node = copy.deepcopy(node)
        node.args = [Add(arg, Constant(offset)) for arg, offset in zip(node.args, self.offset)]
        # print(node)
        return node

def create_block_combiner():
    BLOCK_TYPES = {
        'For': 'body',
        'FunctionDecl': 'defn',
        'MultiNode': 'body'
    }

    FUSIBLE_TYPES = {
        'MultiNode': 'body'
    }

    def make_func(name, attr):
        def visit_method(self, node):
            output = []
            if not getattr(node, attr):
                return node
            for child in getattr(node, attr):
                if child.__class__.__name__ in FUSIBLE_TYPES:
                    child = self.visit(child)
                    output.extend(getattr(child, FUSIBLE_TYPES[child.__class__.__name__]))
                else:
                    output.append(child)
            setattr(node, attr, output)
            return node
        return visit_method

    return type(
        "BlockCombineTransformer", (ast.NodeTransformer,), {
            'visit_{}'.format(block_type) : make_func(block_type, attrib) for block_type, attrib in BLOCK_TYPES.items()
        }
    )

BlockCombineTransformer = create_block_combiner()
del create_block_combiner

def split_stencil(stencil, n):
    """
    Currently only works on direct summing stencils
    :return:
    """
    component = stencil.op_tree
    parts = []
    should_exit = False
    while not should_exit:
        parts.append([])
        for _ in range(n):
            if not isinstance(component, StencilOp) or component.op is not operator.add:
                should_exit = True
                break
            parts[-1].append(component.left)
            component = component.right
    return parts

