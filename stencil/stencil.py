import abc
import ast
import operator
import itertools
from stencil_compiler import StencilCompiler, NameFinder
from vector import Vector

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

    def compile_to_ast(self):
        index_name = 'index'
        nodes = self._compile_to_ast(self.op_tree, index_name=index_name)
        if not isinstance(nodes, (list, tuple)):
            nodes = [nodes]
        nodes[-1] = ast.Return(nodes[-1])

        array_names = NameFinder.get_names(self.op_tree)

        function_name = 'kernel'
        args = [
            index_name,
        ]
        args.extend(sorted(array_names))
        return ast.FunctionDef(
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

    def _get_callable(self):
        tree = self.compile_to_ast()
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
        self.value = value

    def __str__(self):
        return str(self.value)

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
        if not isinstance(arr[0], (list, tuple)):
            return tuple(
                el if isinstance(el, StencilComponent) else StencilConstant(el) for el in arr
            )
        return tuple(cls.componentize(sub_array) for sub_array in arr)

    @classmethod
    def flatten(cls, arr):
        if not isinstance(arr[0], (list, tuple)):
            return arr
        return tuple(itertools.chain.from_iterable(cls.flatten(a) for a in arr))

    def __init__(self, data):
        self.data = self.componentize(data)

    @property
    def weights(self):
        return self.flatten(self.data)

    def __getitem__(self, item):
        if isinstance(item, (list, tuple)):
            obj = self.data
            for s in item:
                obj = obj[s]
            return obj
        return self.data[item]



class StencilOp(StencilNode):
    _fields = ['a', 'b']

    _op_map = {
        operator.add: "+",
        operator.mul: "*",
        operator.div: "/",
        operator.sub: "-"
    }

    def __init__(self, a, b, op):
        self.a = a if isinstance(a, StencilComponent) else StencilConstant(a)
        self.b = b if isinstance(b, StencilComponent) else StencilConstant(b)
        self.op = op

    def __str__(self):
        left = str(self.a)
        right = str(self.b)
        split_left = left.split("\n")
        split_right = right.split("\n")
        num_rows = max(len(split_left), len(split_right))
        left_insert = num_rows - len(split_left)
        right_insert = num_rows - len(split_right)
        pad = lambda split, insert: [""] * (insert / 2) + split + [""] * (insert - insert/2)
        split_left = pad(split_left, left_insert)
        split_right = pad(split_right, right_insert)
        middle = num_rows / 2
        middle_elements = [" " if row_num != middle else self._op_map[self.op] for row_num in range(num_rows)]
        return "\n".join(l + m + r for l, m, r in zip(split_left, middle_elements, split_right))