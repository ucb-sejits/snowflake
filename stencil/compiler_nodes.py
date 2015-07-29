from collections import namedtuple
import copy

__author__ = 'nzhang-dev'

import ast

class StencilCompilerNode(ast.AST):
    pass


class IndexOp(StencilCompilerNode):
    _fields = ['elts', 'ndim', 'name']
    def __init__(self, elts):
        self.elts = elts
        self._name = None

    @property
    def name(self):
        if self._name is not None:
            return self._name

        for elt in self.elts:
            if elt.name is not None:
                self._name = elt.name

        return self._name

    @property
    def ndim(self):
        return len(self.elts)


class ArrayIndex(IndexOp):
    def __init__(self, name, ndim):
        super(ArrayIndex, self).__init__(
            elts=[
                ast.Name(id="{name}_{dim}".format(name=name, dim=dim), ctx=ast.Load()) for dim in range(ndim)
            ]
        )
        self._name = name


class IterationSpace(StencilCompilerNode):
    _fields = ['space', 'body']

    Dimension = namedtuple("Dimension", ['low', 'high', 'stride'])

    def __init__(self, space, body):
        new_space = []
        for dim_range in space:
            if len(dim_range) == 1:
                dim_range = (0, dim_range[0], 1)
            elif len(dim_range) == 2:
                dim_range = (dim_range[0], dim_range[1], 1)
            elif len(dim_range) == 3:
                dim_range = tuple(dim_range)
            else:
                raise ValueError("Invalid arguments for IterationSpace")
            new_space.append(self.Dimension(*dim_range))
        self.space = tuple(new_space)
        self.body = body

    def __deepcopy__(self, memo):
        print("ITERATION SPACE")
        return type(self)(
            copy.deepcopy(self.space, memo=memo),
            copy.deepcopy(self.body, memo=memo)
        )