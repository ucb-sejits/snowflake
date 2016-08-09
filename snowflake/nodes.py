import ast
from collections import Iterable
import copy
from numbers import Number
import operator
import itertools
from vector import Vector

__author__ = 'nzhang-dev'
class StencilNode(ast.AST):
    pass


class Stencil(StencilNode):
    """
    A Stencil is defined by it's operation tree (snowflake components or other things), where it outputs, and the space it iterates over
    """

    _fields = ["op_tree"]

    def __init__(self, op_tree, output, iteration_space, primary_mesh=None):
        # iteration_space is an iterable of (low, high, stride), (low, high), or (high,) tuples
        self.op_tree = op_tree
        self.output = output
        if not isinstance(iteration_space, (RectangularDomain, DomainUnion)):
            iteration_space = RectangularDomain(iteration_space)
        if isinstance(iteration_space, RectangularDomain):
            iteration_space = DomainUnion([iteration_space])
        self.iteration_space = iteration_space
        self.primary_mesh = primary_mesh or output

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.op_tree, memo),
            copy.deepcopy(self.output, memo),
            copy.deepcopy(self.iteration_space, memo),
            copy.deepcopy(self.primary_mesh, memo)
        )

    def __hash__(self):
        return hash((hash(self.op_tree), hash(self.output), hash(self.iteration_space), hash(self.primary_mesh)))

class DomainUnion(StencilNode):
    _fields = ['domains']

    def __init__(self, domains):
        self.domains = domains

    def __add__(self, other):
        if isinstance(other, RectangularDomain):
            return DomainUnion(copy.deepcopy(self.domains) + [other])
        if isinstance(other, DomainUnion):
            return DomainUnion(copy.deepcopy(self.domains) + copy.deepcopy(other.domains))
        return NotImplemented

    def __reduce__(self):
        return (self.__class__, (self.domains,))

    __radd__ = __add__

    def __hash__(self):
        return hash(tuple(hash(i) for i in self.domains))

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.domains, memo))

    def __str__(self):
        return " U ".join(str(i) for i in self.domains)

    def reify(self, shape):
        return DomainUnion([domain.reify(shape) for domain in self.domains])

class RectangularDomain(StencilNode):
    def __init__(self, space):
        lower, upper, stride = zip(*space)
        self.lower = Vector(lower)
        self.stride = Vector(stride)
        self.upper = Vector(upper)
        if not len(self.lower) == len(self.upper) == len(self.stride):
            raise ValueError("Inconsistent dimensionality in domain")

    def __reduce__(self):
        args = tuple((l, h, s) for l, h, s in zip(self.lower, self.stride, self.upper))
        return (self.__class__, (args,))

    def __add__(self, other):
        if not isinstance(other, RectangularDomain):
            return NotImplemented
        return DomainUnion([self, other])

    __radd__ = __add__

    def __hash__(self):
        return hash((self.lower, self.stride, self.upper))

    def __deepcopy__(self, memo):
        return self

    def __str__(self):
        return "<{}, {}, {}>".format(self.lower, self.upper, self.stride)

    def reify(self, shape):
        if len(self.stride) != len(shape):
            raise ValueError("Shape does not has same number of dimensions as domain")
        low = Vector(
            [i + size if i < 0 else i for i, size in zip(self.lower, shape)]
        )

        high = Vector(
            [i + size if i <= 0 else i for i, size in zip(self.upper, shape)]
        )

        # now we clip the high to fit perfectly. For example, 1 to 10 by 3 should really be just 1, 4, 7 (so 1 to 8)
        # so really the upper bound should be upper - ((upper - lower - 1) % stride)
        high = Vector(
            [(h - ((h - l) % s)) if s else l for h, l, s in zip(high, low, self.stride)]
        )

        # since Rectangular domain expects (low, high, stride) tuples in each dimension, we reform the array
        return RectangularDomain([(l, h, s) for l, h, s in zip(low, high, self.stride)])

class StencilComponent(StencilNode):
    """
    A StencilComponent consists of an array name (string) and its associated array of weights (py:func:`weightArray`).
    Array references are assumed to be unique if their string names are unique.
    """
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
        return type(self)(self.name, copy.deepcopy(self.weights, memo))

    def __hash__(self):
        return hash((hash(self.name), hash(self.weights)))

class StencilConstant(StencilNode):
    """
    A normal number/constant.
    """
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

    __rsub__ = lambda x, y: StencilConstant.__sub__(y, x)

    def __mul__(self, other):
        return self.__apply_op(self, other, operator.mul)

    __rmul__ = __mul__

    def __div__(self, other):
        return self.__apply_op(self, other, operator.div)

    __rdiv__ = lambda x, y: StencilConstant.__div__(y, x)

    def __deepcopy__(self, memo):
        return StencilConstant(self.value)

    def __hash__(self):
        return hash(self.value)

    def __nonzero__(self):
        return bool(self.value)


class WeightArray(StencilNode):
    """
    An array of weights, denoted by either numpy matrices or normal nested lists in python. These can be constants or stencilcomponents.
    """
    _fields = ['weights']

    @property
    def indices(self):
        ranges = [range(i) for i in self.shape]
        return (
            Vector(coords) for coords in itertools.product(*ranges)
            if self[coords]
        )

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
                el if isinstance(el, StencilNode) else StencilConstant(el) for el in arr
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

    def __deepcopy__(self, memo):
        return WeightArray(self.data)

    def __hash__(self):
        return hash((tuple(self.weights), tuple(self.vectors)))


class SparseWeightArray(StencilNode):
    """
    A weight array that's instead defined by a dictionary of vectors relative to the center with their associated values.
    """
    _fields = ['weights']

    def __init__(self, weight_map):
        """
        :param data_list: a dict of point: values
        :return:
        """
        self.__weight_map = {}
        for coord, value in weight_map.items():
            self.__ndim = len(coord)
            if not value:
                continue
            self.__weight_map[Vector(coord)] = value if isinstance(value, StencilNode) else StencilConstant(value)
        #self.__weight_map = weight_map
        self.__key_value_pairs = tuple(self.__weight_map.items())

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

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.__weight_map, memo))

    def __hash__(self):
        return hash((tuple(self.weights), hash(tuple(self.vectors))))

class StencilOp(StencilNode):
    """
    An operation between snowflake nodes. Will also be created as the result of mathematical operations on StencilComponents
    """
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

    def __deepcopy__(self, memo):
        return type(self)(
            copy.deepcopy(self.left, memo),
            copy.deepcopy(self.right, memo),
            self.op
        )

    def __hash__(self):
        return hash((self._op_map[self.op], hash(self.left), hash(self.right)))

class StencilGroup(StencilNode):
    _fields = ['body']

    def __init__(self, body):
        self.body = body
        super(StencilGroup, self).__init__()

    def __deepcopy__(self, memo):
        return type(self)(copy.deepcopy(self.body, memo))

    def __hash__(self):
        return hash(tuple(hash(i) for i in self.body))