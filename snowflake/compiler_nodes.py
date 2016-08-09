from collections import namedtuple
import copy

__author__ = 'nzhang-dev'

import ast
from vector import Vector

class StencilCompilerNode(ast.AST):
    """
    Generic Parent class or Stencil Nodes
    """

    def __init__(self, *args, **kwargs):
        if args and kwargs:
            raise ValueError("Pass either args or kwargs")
        iterable = zip(self._fields, args) if args else kwargs.items()
        for key, value in iterable:
            setattr(self, key, value)


    def __deepcopy__(self, memo):
        print('Copying: ' + str(type(self).__name__))
        raise NotImplementedError("{} does not have __deepcopy__ implemented".format(type(self).__name__))

    def _default_deepcopy(self, memo):
        attribs = [copy.deepcopy(getattr(self, name), memo) for name in self._fields]
        return type(self)(*attribs)


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
    #
    # def __init__(self, space, body):
    #     self.space = space
    #     self.body = body

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.space, memo=memo),
            copy.deepcopy(self.body, memo=memo)
        )

class NDSpace(StencilCompilerNode):

    _fields = ['spaces']
    # def __init__(self, spaces):
    #     self.spaces = spaces

    def __deepcopy__(self, memo):
        return NDSpace(copy.deepcopy(self.spaces, memo))

    @property
    def ndim(self):
        if not self.spaces:
            raise ValueError("Empty Space has dimension")
        return self.spaces[0].ndim

class Block(StencilCompilerNode):
    _fields = ['body']
    # def __init__(self, body):
    #     self.body = body

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.body, memo))

class Space(StencilCompilerNode):
    _fields = ['low', 'high', 'stride']

    __deepcopy__ = StencilCompilerNode._default_deepcopy

    @property
    def ndim(self):
        return len(self.low)



class NestedSpace(StencilCompilerNode):
    """
    Used for tiling/blocking/iteration order manipulation. This frequently requires different/more complex code generation.
    This node requires the space to have reified bounds
    """
    _fields = ['low', 'high', 'block_size', 'stride']

    __deepcopy__ = StencilCompilerNode._default_deepcopy
