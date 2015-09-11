from collections import namedtuple
import copy

__author__ = 'nzhang-dev'

import ast

class StencilCompilerNode(ast.AST):
    """
    Generic Parent class or Stencil Nodes
    """


    def __deepcopy__(self, memo):
        print('Copying: ' + str(type(self).__name__))
        raise NotImplementedError("{} does not have __deepcopy__ implemented".format(type(self).__name__))


class IndexOp(StencilCompilerNode):
    """
    Operation on an index
    """
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

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.elts, memo))


class ArrayIndex(IndexOp):
    """
    Semantic node for an array
    """
    def __init__(self, name, ndim):
        super(ArrayIndex, self).__init__(
            elts=[
                ast.Name(id="{name}_{dim}".format(name=name, dim=dim), ctx=ast.Load()) for dim in range(ndim)
            ]
        )
        self._name = name

    def __deepcopy__(self, memo):
        return type(self)(self.name, self.ndim)


class IterationSpace(StencilCompilerNode):
    """
    Semantic node for the space over which a snowflake is applied.
    """
    _fields = ['space', 'body']

    Dimension = namedtuple("Dimension", ['low', 'high', 'stride'])

    def __init__(self, space, body):
        new_space = []
        for dim_range in space:
            if len(dim_range) == 1:
                dim_range = (0, dim_range[0], None)
            elif len(dim_range) == 2:
                dim_range = (dim_range[0], dim_range[1], None)
            elif len(dim_range) == 3:
                dim_range = tuple(dim_range)
            else:
                raise ValueError("Invalid arguments for IterationSpace")
            new_space.append(self.Dimension(*dim_range))
        self.space = tuple(new_space)
        self.body = body

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.space, memo=memo),
            copy.deepcopy(self.body, memo=memo)
        )

class Block(StencilCompilerNode):
    _fields = ['body']
    def __init__(self, body):
        self.body = body

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.body, memo))