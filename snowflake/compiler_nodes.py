from collections import namedtuple
import copy

__author__ = 'nzhang-dev'

import ast

Space = namedtuple("Space", ['low', 'high', 'stride'])

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

    def __init__(self, space, body):
        self.space = space
        self.body = body

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.space, memo=memo),
            copy.deepcopy(self.body, memo=memo)
        )

class SpaceUnion(StencilCompilerNode):

    _fields = ['spaces']
    def __init__(self, spaces):
        self.spaces = spaces

    def __deepcopy__(self, memo):
        return SpaceUnion(copy.deepcopy(self.spaces, memo))

class Block(StencilCompilerNode):
    _fields = ['body']
    def __init__(self, body):
        self.body = body

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.body, memo))