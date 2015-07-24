import abc
import ast
from collections import Iterable
import operator
import itertools
from stencil_compiler import StencilCompiler, find_names
from vector import Vector
import numpy as np

__author__ = 'nzhang-dev'

###############################################################################################################
# A component is defined by a weight array, which may or may not recursively contain components.
###############################################################################################################


class Stencil(object):

    @staticmethod
    def _compile_to_ast(node, index_name):
        return StencilCompiler(index_name).visit(node)

    def __init__(self, op_tree):
        self.op_tree = op_tree

    @property
    def names(self):
        return {node.name for node in ast.walk(self.op_tree) if hasattr(node, "name")}

    def compile_to_ast(self, index_name):
        nodes = self._compile_to_ast(self.op_tree, index_name=index_name)
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        return nodes

    def get_kernel(self):
        index_name = 'index'
        nodes = self.compile_to_ast(index_name)
        nodes[-1] = ast.Return(nodes[-1])

        array_names = find_names(self.op_tree)

        function_name = 'kernel'
        args = [
            index_name,
        ]
        args.extend(sorted(array_names))
        tree = ast.FunctionDef(
            name=function_name,
            args=ast.arguments(
                args=[ast.Name(id=arg, ctx=ast.Param()) for arg in args],
                vararg=None,
                kwarg=None,
                defaults=[]
            ),
            body=nodes,
            decorator_list=[]
        )
        tree = ast.Module(body=[tree])
        tree = ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'exec')
        exec code in globals(), locals()
        return locals()['kernel']





class StencilNode(ast.AST):
    pass


class StencilComponent(StencilNode):
    _fields = ["weights"]

    def __init__(self, name, weights):
        self.name = name
        assert isinstance(weights, WeightArray)
        self.weights = weights

    def __add__(self, other):
        return StencilOp(self, other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return StencilOp(self, other, operator.sub)

    __rsub__ = lambda x, y: StencilComponent.__sub__(y, x)

    def __mul__(self, other):
        return StencilOp(self, other, operator.mul)

    __rmul__ = __mul__

    def __div__(self, other):
        return StencilOp(self, other, operator.div)

    __rdiv__ = lambda x, y: StencilComponent.__div__(y, x)

    def __getitem__(self, item):
        return self.weights[item]


class StencilConstant(StencilNode):

    def __init__(self, value):
        # while isinstance(value, StencilConstant):
        #     value = value.value
        self.value = value

    @classmethod
    def __apply_op(cls, a, b, op):
        if isinstance(a, (int, float)):
            a = cls(a)
        if isinstance(b, (int, float)):
            b = cls(b)
        if isinstance(a, cls) and isinstance(b, cls):
            return cls(op(a.value, b.value))
        return StencilOp(a, b, op)

    def __add__(self, other):
        return self.__apply_op(self, other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.__apply_op(self, other, operator.sub)

    __rsub__ = lambda x, y: StencilComponent.__sub__(y, x)

    def __mul__(self, other):
        return self.__apply_op(self, other, operator.mul)

    __rmul__ = __mul__

    def __div__(self, other):
        return self.__apply_op(self, other, operator.div)

    __rdiv__ = lambda x, y: StencilComponent.__div__(y, x)


class WeightArray(StencilNode):
    _fields = ['weights']

    @property
    def radius(self):
        return self.shape / 2

    @property
    def indices(self):
        ranges = [range(i) for i in self.shape]
        return (Vector(coords) for coords in itertools.product(*ranges))

    @property
    def center(self):
        return self.radius

    @property
    def vectors(self):
        return (index - self.center for index in self.indices)

    @classmethod
    def _get_shape(cls, arr):
        if not isinstance(arr[0], (list, tuple)):
            return (len(arr),)
        return (len(arr),) + cls._get_shape(arr[0])

    @property
    def shape(self):
        return Vector(self._get_shape(self.data))

    @classmethod
    def componentize(cls, arr):
        if not isinstance(arr[0], Iterable):
            return [
                el if isinstance(el, StencilComponent) else StencilConstant(el) for el in arr
            ]
        return [cls.componentize(sub_array) for sub_array in arr]

    @classmethod
    def flatten(cls, arr):
        if not isinstance(arr[0], Iterable):
            return arr
        return tuple(itertools.chain.from_iterable(cls.flatten(a) for a in arr))

    def __init__(self, data):
        self.data = self.componentize(data)

    @property
    def weights(self):
        return list(self.flatten(self.data))

    def __getitem__(self, item):
        if isinstance(item, Iterable):
            obj = self.data
            for s in item:
                obj = obj[s]
            return obj
        return self.data[item]

    def __setitem__(self, key, value):
        if isinstance(key, Iterable):
            obj = self.data
            for s in key[:-1]:
                obj = obj[s]
            obj[key[-1]] = value
        self.data[key] = value

class StencilOp(StencilNode):
    _fields = ['left', 'right']

    _op_map = {
        operator.add: "+",
        operator.mul: "*",
        operator.div: "/",
        operator.sub: "-"
    }

    def __init__(self, left, right, op):
        # self.left = left if isinstance(left, StencilComponent) else StencilConstant(left)
        # self.right = right if isinstance(right, StencilComponent) else StencilConstant(right)
        self.left = StencilConstant(left) if isinstance(left, (int, float)) else left
        self.right = StencilConstant(right) if isinstance(right, (int, float)) else right
        self.op = op


    @classmethod
    def __apply_op(cls, a, b, op):
        if isinstance(a, (int, float)):
            a = StencilConstant(a)
        if isinstance(b, (int, float)):
            b = StencilConstant(b)
        return cls(a, b, op)

    def __add__(self, other):
        return self.__apply_op(self, other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self.__apply_op(self, other, operator.sub)

    __rsub__ = lambda x, y: StencilOp.__sub__(y, x)

    def __mul__(self, other):
        return self.__apply_op(self, other, operator.mul)

    __rmul__ = __mul__

    def __div__(self, other):
        return self.__apply_op(self, other, operator.div)

    __rdiv__ = lambda x, y: StencilOp.__div__(y, x)