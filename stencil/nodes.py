import ast
from collections import Iterable
import copy
import operator
import itertools
from vector import Vector

__author__ = 'nzhang-dev'
class StencilNode(ast.AST):
    pass


class Stencil(StencilNode):

    _fields = ["op_tree", "output", "iteration_space", "total_shape"]

    def __init__(self, op_tree, output, iteration_space):
        # iteration_space is an iterable of (low, high, stride), (low, high), or (high,) tuples
        self.op_tree = op_tree
        self.output = output
        self.iteration_space = iteration_space



class StencilComponent(StencilNode):
    _fields = ["name", "weights"]

    def __init__(self, name, weights):
        self.name = name
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

    def __deepcopy__(self, memo):
        print('deepcopy', memo)
        return self

    def __copy__(self):
        print('copy')
        return self


class StencilConstant(StencilNode):

    _fields = ['value']

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
    def indices(self):
        ranges = [range(i) for i in self.shape]
        return (Vector(coords) for coords in itertools.product(*ranges))

    @property
    def center(self):
        return self.shape / 2

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
    def __componentize(cls, arr):
        if not isinstance(arr[0], Iterable):
            return [
                el if isinstance(el, StencilComponent) else StencilConstant(el) for el in arr
            ]
        return [cls.__componentize(sub_array) for sub_array in arr]

    @classmethod
    def _flatten(cls, arr):
        if not isinstance(arr[0], Iterable):
            return arr
        return tuple(itertools.chain.from_iterable(cls._flatten(a) for a in arr))

    def __init__(self, data):
        self.data = self.__componentize(data)

    @property
    def weights(self):
        return list(self._flatten(self.data))

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


class SparseWeightArray(StencilNode):
    _fields = ['weights']

    def __init__(self, weight_map):
        """
        :param data_list: a dict of point: values
        :return:
        """
        self.__weight_map = {}
        for coord, value in weight_map.items():
            self.__weight_map[Vector(coord)] = value if isinstance(value, StencilNode) else StencilConstant(value)
        #self.__weight_map = weight_map
        self.__key_value_pairs = tuple(self.__weight_map.items())
        self.__ndim = len(self.__key_value_pairs[0][0])

    @property
    def weights(self):
        return [value for key, value in self.__key_value_pairs]

    @property
    def indices(self):
        return [key for key, value in self.__key_value_pairs]

    @property
    def vectors(self):
        return [key for key, value in self.__key_value_pairs]

    @property
    def center(self):
        return Vector((0,) * self.__ndim)

    def __getitem__(self, item):
        return self.__weight_map[item]

    def __setitem__(self, key, value):
        self.__weight_map[key] = value


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


class RangeNode(StencilNode):
    _fields = ['target', 'body']
    def __init__(self, target, iterator, body):
        self.iterator = iterator
        self.target = target
        self.body = body

    def __deepcopy__(self, memo):
        return RangeNode(self.target, self.iterator, copy.deepcopy(self.body))

class StencilBlock(StencilNode):
    _fields = ['body']
    def __init__(self, body):
        self.body = body