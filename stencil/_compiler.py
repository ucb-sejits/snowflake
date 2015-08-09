import ast
import functools
import operator

from compiler_nodes import ArrayIndex, IndexOp, IterationSpace
from nodes import StencilComponent, StencilConstant
from nodes import Stencil


__author__ = 'nzhang-dev'


class StencilCompiler(ast.NodeVisitor):

    operator_to_ast_map = {
        operator.add: ast.Add(),
        operator.sub: ast.Sub(),
        operator.mul: ast.Mult(),
        operator.div: ast.Div()
    }

    def __init__(self, index_name='index', ndim=0):
        self.index_name = index_name
        self.ndim = ndim

    def visit_StencilBlock(self, node):
        return [self.visit(i) for i in node.body]

    def visit_Stencil(self, node):
        body = self.visit(node.op_tree)
        assignment = ast.Assign(
            targets=[
                ast.Subscript(
                    value=ast.Name(id=node.output, ctx=ast.Load()),
                    slice=ast.Index(ast.Name(id=self.index_name, ctx=ast.Load())),
                    ctx=ast.Store()
                )
            ],
            value=body
        )
        nested = IterationSpace(space=node.iteration_space, body=[assignment])
        return nested


    def visit_StencilConstant(self, node):
        return ast.Num(n=node.value, ctx=ast.Load())

    def visit_StencilComponent(self, node):
        weights = [self.visit(weight) for weight in node.weights.weights]
        components = [
            ast.BinOp(
                left=ast.Subscript(
                    value=ast.Name(
                        id=node.name,
                        ctx=ast.Load()
                    ),
                    slice=ast.Index(
                        value=ast.BinOp(
                            left=ast.Name(id=self.index_name, ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Tuple(
                                elts=[ast.Num(n=i) for i in vector],
                                ctx=ast.Load()
                            )
                        )
                    ),
                    ctx=ast.Load()
                ),
                op=ast.Mult(),
                right=weight
            ) for weight, vector in zip(weights, node.weights.vectors)
            if (not isinstance(weight, StencilConstant)) or weight.value != 0
        ]
        if not components:  # we filtered all of it out
            return ast.Num(n=0)
        return functools.reduce(
            lambda x, y: ast.BinOp(left=x, right=y, op=ast.Add()),
            components
        )

    def visit_StencilOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = self.operator_to_ast_map[node.op]
        return ast.BinOp(left=left, op=op, right=right)


class ArrayOpRecognizer(ast.NodeTransformer):
    AST_to_op = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.div
    }

    def __init__(self, index_name, ndim):
        self.index_name = index_name
        self.ndim = ndim

    def visit_Name(self, node):
        if node.id == self.index_name:
            return ArrayIndex(node.id, self.ndim)
        return node

    def visit_BinOp(self, node):
        node.left = self.visit(node.left)
        node.right = self.visit(node.right)
        if isinstance(node.left, IndexOp) or isinstance(node.right, IndexOp):
            elts = []
            for index, (left, right) in enumerate(zip(node.left.elts, node.right.elts)):
                elts.append(
                    ast.BinOp(left=left, op=node.op, right=right)
                )
            return IndexOp(elts)
        return node


class OpSimplifier(ast.NodeTransformer):
    rules = {
        ast.Add: {(0, 'right'): 'right', ('left', 0): 'left'},
        ast.Mult: {(0, 'right'): 0, ('left', 0): 0, (1, 'right'): 'right', ('left', 1): 'left'}
    }

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return self.get_result(node)

    @classmethod
    def get_result(cls, node):
        left = node.left
        right = node.right
        op = node.op
        patterns = cls.rules.get(type(op), {}).items()
        for pattern, result in patterns:
            if cls.__matches(left, right, pattern):
                #print("simplified")
                return cls.__apply_result(left, right, result)

        return node


    @staticmethod
    def __matches(left, right, pattern):
        left_match = right_match = False
        if pattern[0] == 'left':
            left_match = True
        elif isinstance(left, ast.Num):
            if isinstance(pattern[0], (int, float)):
                left_match = pattern[0] == left.n
        if pattern[1] == 'right':
            right_match = True
        elif isinstance(right, ast.Num):
            if isinstance(pattern[1], (int, float)):
                right_match = pattern[1] == right.n
        return left_match and right_match

    @staticmethod
    def __apply_result(left, right, result):
        if result == 'left':
            return left
        elif result == 'right':
            return right
        if isinstance(result, (int, float)):
            return ast.Num(n=result)



class DependencyAnalyzer(ast.NodeVisitor):

    def visit_StencilComponent(self, node):
        dependencies = self.generic_visit(node)
        name = node.name
        node_dependencies = (vector for vector, index in zip(node.vectors, node.indices)
                             if not (isinstance(node, StencilConstant) and node[index].value == 0))


def find_names(node):
    names = set()
    target = None
    # if hasattr(node, 'name'):
    #     names.add(node.name)
    class Visitor(ast.NodeVisitor):
        def visit_StencilComponent(self, node):
            names.add(node.name)
            self.generic_visit(node)
        def visit_Stencil(self, node):
            names.add(node.output)
            self.generic_visit(node)
    # for n in ast.walk(node):
    #     if isinstance(n, StencilComponent):
    #         #print("found")
    #         names.add(n.name)
    #     elif isinstance(n, Stencil):
    #         names.add(n.output)
    Visitor().visit(node)
    return names

