import ast
import functools
import operator
from stencil import WeightArray, StencilComponent, StencilConstant, SparseWeightArray

__author__ = 'nzhang-dev'


class StencilCompiler(ast.NodeVisitor):

    operator_to_ast_map = {
        operator.add: ast.Add(),
        operator.sub: ast.Sub(),
        operator.mul: ast.Mult(),
        operator.div: ast.Div()
    }

    def __init__(self, index_name='index'):
        self.index_name = index_name

    def visit_Stencil(self, node):
        return self.visit(node.op_tree)

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
            if not (isinstance(weight, StencilConstant) and weight.value == 0)
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

class DependencyAnalyzer(ast.NodeVisitor):

    def visit_StencilComponent(self, node):
        dependencies = self.generic_visit(node)
        name = node.name
        node_dependencies = (vector for vector, index in zip(node.vectors, node.indices)
                             if not (isinstance(node, StencilConstant) and node[index].value == 0))


def find_names(node):
    names = set()
    # if hasattr(node, 'name'):
    #     names.add(node.name)
    for n in ast.walk(node):
        if isinstance(n, StencilComponent):
            #print("found")
            names.add(n.name)
    return names